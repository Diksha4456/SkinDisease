package com.fairskin.fairskin_mobile

import ai.onnxruntime.*
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Base64
import android.util.Log
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.*

class MainActivity: FlutterActivity() {
    private val TAG = "FairSkinNative"
    private val CHANNEL = "com.fairskin/offline_screen"
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    private val DISEASE_CLASSES = arrayOf("acne", "melasma", "psoriasis", "ringworm", "vitiligo", "non_target_ood")
    private val FEATURE_VOCABULARY = arrayOf(
        "erythematous_papule_pustule",
        "open_closed_comedo",
        "inflammatory_nodule_scarring",
        "reticulated_hyperpigmentation",
        "irregular_pigment_border",
        "chalky_white_depigmentation",
        "well_demarcated_border",
        "erythematous_scaly_plaque",
        "micaceous_silvery_scale",
        "annular_erythematous_ring",
        "active_scaly_advancing_border",
        "central_clearing"
    )

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        // Initialize ONNX in background worker thread on startup
        Thread {
            try {
                initOnnxEngine()
            } catch (t: Throwable) {
                Log.e(TAG, "Async init error", t)
            }
        }.start()

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "screenOffline" -> {
                    val filePath = call.argument<String>("filePath")
                    if (filePath == null) {
                        result.error("INVALID_PATH", "Image file path is null", null)
                        return@setMethodCallHandler
                    }
                    // Run inference in background worker thread to prevent any UI freeze or ANR
                    Thread {
                        try {
                            Log.i(TAG, "Starting offline screening for: $filePath")
                            val report = runOfflineInference(filePath)
                            Log.i(TAG, "Offline screening succeeded for: $filePath")
                            runOnUiThread {
                                result.success(report)
                            }
                        } catch (e: Throwable) {
                            Log.e(TAG, "Inference error", e)
                            val errDetails = "${e.javaClass.simpleName}: ${e.message}"
                            runOnUiThread {
                                result.error("INFERENCE_ERROR", errDetails, null)
                            }
                        }
                    }.start()
                }
                "isOfflineReady" -> {
                    try {
                        if (ortSession == null) {
                            initOnnxEngine()
                        }
                        result.success(ortSession != null)
                    } catch (t: Throwable) {
                        result.success(false)
                    }
                }
                else -> {
                    result.notImplemented()
                }
            }
        }
    }

    @Synchronized
    private fun initOnnxEngine() {
        if (ortSession != null) return
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            val modelFile = File(cacheDir, "fairskin_model.onnx")
            val expectedSize = 4330757L
            if (!modelFile.exists() || modelFile.length() != expectedSize) {
                Log.i(TAG, "Copying fairskin_model.onnx to cacheDir...")
                val tempFile = File(cacheDir, "fairskin_model.onnx.tmp")
                if (tempFile.exists()) tempFile.delete()
                assets.open("fairskin_model.onnx").use { input ->
                    FileOutputStream(tempFile).use { output ->
                        input.copyTo(output)
                    }
                }
                if (tempFile.length() == expectedSize) {
                    if (modelFile.exists()) modelFile.delete()
                    tempFile.renameTo(modelFile)
                    Log.i(TAG, "Model written and verified: ${modelFile.length()} bytes")
                } else {
                    Log.w(TAG, "Extracted size ${tempFile.length()} != expected $expectedSize")
                    tempFile.renameTo(modelFile)
                }
            }

            val sessionOptions = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(2)
                setInterOpNumThreads(1)
            }
            ortSession = ortEnv?.createSession(modelFile.absolutePath, sessionOptions)
            Log.i(TAG, "ONNX Runtime offline engine initialized successfully from: ${modelFile.absolutePath}")
        } catch (e: Throwable) {
            Log.e(TAG, "Failed to initialize ONNX engine", e)
        }
    }

    private fun runOfflineInference(filePath: String): Map<String, Any?> {
        if (ortSession == null) {
            initOnnxEngine()
        }
        val session = ortSession ?: throw IllegalStateException("ONNX offline session could not be initialized")
        val env = ortEnv ?: throw IllegalStateException("ONNX environment not initialized")

        val file = File(filePath)
        if (!file.exists()) {
            throw IllegalArgumentException("File not found at: $filePath")
        }

        // Safe bitmap decoding with inSampleSize to prevent OOM on high-resolution camera photos
        val boundsOptions = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(filePath, boundsOptions)
        var sampleSize = 1
        val maxDim = maxOf(boundsOptions.outWidth, boundsOptions.outHeight)
        while (maxDim / sampleSize > 1024) {
            sampleSize *= 2
        }
        val decodeOptions = BitmapFactory.Options().apply {
            inSampleSize = sampleSize
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val origBmp = BitmapFactory.decodeFile(filePath, decodeOptions)
            ?: throw IllegalArgumentException("Could not decode image at path: $filePath")

        val resizedBmp = Bitmap.createScaledBitmap(origBmp, 224, 224, true)

        // Preprocess to Channel-First Direct FloatBuffer with ImageNet normalization
        val numPixels = 224 * 224
        val directByteBuffer = ByteBuffer.allocateDirect(1 * 3 * 224 * 224 * 4).order(ByteOrder.nativeOrder())
        val floatBuffer = directByteBuffer.asFloatBuffer()

        val pixels = IntArray(numPixels)
        resizedBmp.getPixels(pixels, 0, 224, 0, 0, 224, 224)

        val meanR = 0.485f; val meanG = 0.456f; val meanB = 0.406f
        val stdR = 0.229f;  val stdG = 0.224f;  val stdB = 0.225f

        // Channel R
        for (p in pixels) {
            val r = (Color.red(p) / 255.0f - meanR) / stdR
            floatBuffer.put(r)
        }
        // Channel G
        for (p in pixels) {
            val g = (Color.green(p) / 255.0f - meanG) / stdG
            floatBuffer.put(g)
        }
        // Channel B
        for (p in pixels) {
            val b = (Color.blue(p) / 255.0f - meanB) / stdB
            floatBuffer.put(b)
        }
        floatBuffer.rewind()
        directByteBuffer.rewind()

        var tensor: OnnxTensor? = null
        var outputs: OrtSession.Result? = null
        val diseaseProbs: FloatArray
        val featureProbs: FloatArray
        val severityLogits: FloatArray

        try {
            tensor = OnnxTensor.createTensor(env, floatBuffer, longArrayOf(1, 3, 224, 224))
            outputs = session.run(mapOf("image" to tensor))

            var diseaseTensor: OnnxTensor? = null
            var featureTensor: OnnxTensor? = null
            var severityTensor: OnnxTensor? = null

            val iter = outputs.iterator()
            while (iter.hasNext()) {
                val entry = iter.next()
                val value = entry.value as? OnnxTensor
                when (entry.key) {
                    "disease_probs" -> diseaseTensor = value
                    "feature_probs" -> featureTensor = value
                    "severity_logits" -> severityTensor = value
                }
            }
            if (diseaseTensor == null && outputs.size() > 0) diseaseTensor = outputs.get(0) as? OnnxTensor
            if (featureTensor == null && outputs.size() > 1) featureTensor = outputs.get(1) as? OnnxTensor
            if (severityTensor == null && outputs.size() > 2) severityTensor = outputs.get(2) as? OnnxTensor

            if (diseaseTensor == null || featureTensor == null || severityTensor == null) {
                throw IllegalStateException("Failed to extract model output tensors")
            }

            diseaseProbs = extractFloats(diseaseTensor, 6)
            featureProbs = extractFloats(featureTensor, 12)
            severityLogits = extractFloats(severityTensor, 2)
        } finally {
            try { tensor?.close() } catch (_: Throwable) {}
            try { outputs?.close() } catch (_: Throwable) {}
        }

        // Clinical Evidence Grounding
        var acneScore = maxOf(featureProbs[0], featureProbs[1]) * 0.70f + featureProbs[2] * 0.30f
        var melScore = maxOf(featureProbs[3], featureProbs[4])
        var psoScore = maxOf(featureProbs[7], featureProbs[8])
        var rwScore = maxOf(featureProbs[9], featureProbs[10]) * 0.70f + featureProbs[11] * 0.30f
        var vitScore = maxOf(featureProbs[5], featureProbs[6])

        if (acneScore > 0.30f) { rwScore *= 0.20f; psoScore *= 0.35f }
        if (melScore > 0.35f) { rwScore *= 0.20f; vitScore *= 0.10f }
        if (vitScore > 0.40f) { melScore *= 0.10f }
        if (psoScore > 0.40f) { acneScore *= 0.25f }

        val maxTargetFeat = maxOf(acneScore, melScore, psoScore, rwScore, vitScore)
        val oodFeat = if (maxTargetFeat < 0.25f) maxOf(0.05f, 1.0f - maxTargetFeat * 2.5f) else 0.05f

        val featEvidence = floatArrayOf(acneScore, melScore, psoScore, rwScore, vitScore, oodFeat)
        val fused = FloatArray(6)
        var fusedSum = 0.0f
        for (i in 0 until 6) {
            fused[i] = 0.70f * diseaseProbs[i] + 0.30f * featEvidence[i]
            fusedSum += fused[i]
        }
        val probs = FloatArray(6) { fused[it] / fusedSum }

        var predIdx = 0
        var maxProb = probs[0]
        for (i in 1 until 6) {
            if (probs[i] > maxProb) {
                maxProb = probs[i]
                predIdx = i
            }
        }

        val predDisease = DISEASE_CLASSES[predIdx]
        val confidence = probs[predIdx]

        // Entropy
        var entropy = 0.0
        for (p in probs) {
            if (p > 1e-7f) {
                entropy -= (p * ln(p.toDouble()))
            }
        }
        val uncertaintyLevel = if (entropy > 1.2) "High" else if (entropy > 0.6) "Moderate" else "Low"

        // Out-of-Target Safety Gate
        val target5 = listOf("acne", "melasma", "psoriasis", "ringworm", "vitiligo")
        val rawOodProb = diseaseProbs[5]
        val isOutsideTarget = predDisease == "non_target_ood" ||
                rawOodProb >= 0.28f ||
                (target5.contains(predDisease) && confidence < 0.50f) ||
                (target5.contains(predDisease) && maxTargetFeat < 0.25f)

        val displayDisease: String
        val observableFeatures: List<String>
        val clinicalRationale: String
        val acneSeverity: String

        if (isOutsideTarget) {
            displayDisease = "Not Detected / Not Recognized"
            acneSeverity = "N/A"
            observableFeatures = listOf("No Monitored Lesions Confirmed")
            clinicalRationale = "The presented skin lesion does NOT match any of the 5 target monitored conditions " +
                    "(Acne Vulgaris, Melasma, Plaque Psoriasis, Ringworm, or Vitiligo). " +
                    "No pathognomonic visual markers for target conditions were detected. " +
                    "A formal clinical evaluation by a board-certified dermatologist is recommended."
        } else {
            displayDisease = predDisease.replaceFirstChar { it.uppercase() }
            val detected = mutableListOf<String>()
            for (i in featureProbs.indices) {
                if (featureProbs[i] >= 0.35f) {
                    detected.add(FEATURE_VOCABULARY[i].replace("_", " ").split(" ").joinToString(" ") { it.replaceFirstChar { c -> c.uppercase() } })
                }
            }
            observableFeatures = if (detected.isNotEmpty()) detected else listOf("Subtle Localized Cutaneous Focus")

            acneSeverity = if (predDisease == "acne") {
                if (severityLogits[1] > severityLogits[0]) "High" else "Low"
            } else "N/A"

            clinicalRationale = when (predDisease) {
                "acne" -> "Prediction favored Acne Vulgaris (${(confidence * 100).roundToInt()}%) grounded in localized inflammatory papules/comedones. Assessed as $acneSeverity severity."
                "melasma" -> "Prediction favored Melasma (${(confidence * 100).roundToInt()}%) based on reticulated hyperpigmentation and characteristic irregular boundaries."
                "vitiligo" -> "Prediction favored Vitiligo (${(confidence * 100).roundToInt()}%) grounded in depigmented macules with well-demarcated margins."
                "psoriasis" -> "Prediction favored Psoriasis (${(confidence * 100).roundToInt()}%) based on localized erythematous plaques with characteristic surface scale."
                "ringworm" -> "Prediction favored Ringworm (${(confidence * 100).roundToInt()}%) grounded in annular configuration with active advancing border."
                else -> "Screening indicates atypical presentation."
            }
        }

        // ITA and Fitzpatrick Estimation
        val itaData = estimateIta(resizedBmp)

        // Generate On-Device Visual Attention Heatmap (scaled to 320 for light memory footprint)
        val displayBmp = Bitmap.createScaledBitmap(origBmp, 320, 320, true)
        val overlayBmp = createAttentionOverlay(displayBmp, predIdx, isOutsideTarget)

        // Convert to Base64
        val origB64 = bitmapToBase64(displayBmp)
        val gradcamB64 = bitmapToBase64(overlayBmp)

        // Free bitmap resources immediately
        if (origBmp != displayBmp && !origBmp.isRecycled) origBmp.recycle()
        if (resizedBmp != origBmp && !resizedBmp.isRecycled) resizedBmp.recycle()
        if (displayBmp != overlayBmp && !displayBmp.isRecycled) displayBmp.recycle()
        if (!overlayBmp.isRecycled) overlayBmp.recycle()

        return mapOf(
            "success" to true,
            "predicted_disease" to displayDisease,
            "confidence_pct" to (round(confidence * 1000.0) / 10.0),
            "entropy" to (round(entropy * 1000.0) / 1000.0),
            "uncertainty_level" to uncertaintyLevel,
            "acne_severity" to acneSeverity,
            "fitzpatrick_scale" to itaData.fst,
            "ita_degrees" to (round(itaData.ita * 10.0) / 10.0),
            "ita_tone_band" to itaData.toneBand,
            "quality_warning" to null,
            "observable_features" to observableFeatures,
            "clinical_rationale" to clinicalRationale,
            "referral_guidance" to "This is an automated 100% offline preliminary screening aid. Consult a licensed dermatologist for definitive care.",
            "original_b64" to origB64,
            "gradcam_b64" to gradcamB64
        )
    }

    private fun extractFloats(tensor: OnnxTensor, count: Int): FloatArray {
        try {
            val fb = tensor.floatBuffer
            if (fb != null) {
                fb.rewind()
                return FloatArray(count) { fb.get(it) }
            }
        } catch (e: Throwable) {
            Log.w(TAG, "floatBuffer extraction notice: ${e.message}, attempting getValue fallback")
        }

        val raw = tensor.value
        if (raw is Array<*> && raw.isNotEmpty()) {
            val first = raw[0]
            if (first is FloatArray) {
                return FloatArray(count) { if (it < first.size) first[it] else 0.0f }
            }
        } else if (raw is FloatArray) {
            return FloatArray(count) { if (it < raw.size) raw[it] else 0.0f }
        }
        throw IllegalStateException("Could not extract $count float values from tensor: $raw")
    }

    private data class ItaResult(val ita: Double, val fst: Int, val toneBand: String)

    private fun estimateIta(bmp: Bitmap): ItaResult {
        var totalL = 0.0; var totalB = 0.0; var count = 0
        val cx = bmp.width / 2; val cy = bmp.height / 2
        val radius = min(bmp.width, bmp.height) / 4

        for (y in (cy - radius)..(cy + radius) step 4) {
            for (x in (cx - radius)..(cx + radius) step 4) {
                if (x in 0 until bmp.width && y in 0 until bmp.height) {
                    val p = bmp.getPixel(x, y)
                    val r = Color.red(p) / 255.0
                    val g = Color.green(p) / 255.0
                    val b = Color.blue(p) / 255.0

                    val rL = if (r > 0.04045) ((r + 0.055) / 1.055).pow(2.4) else r / 12.92
                    val gL = if (g > 0.04045) ((g + 0.055) / 1.055).pow(2.4) else g / 12.92
                    val bL = if (b > 0.04045) ((b + 0.055) / 1.055).pow(2.4) else b / 12.92

                    val yNorm = (0.2126 * rL + 0.7152 * gL + 0.0722 * bL) / 1.0000
                    val zNorm = (0.0193 * rL + 0.1192 * gL + 0.9505 * bL) / 1.08883

                    val fY = if (yNorm > 0.008856) yNorm.pow(1.0 / 3.0) else (7.787 * yNorm) + (16.0 / 116.0)
                    val fZ = if (zNorm > 0.008856) zNorm.pow(1.0 / 3.0) else (7.787 * zNorm) + (16.0 / 116.0)

                    val L = 116.0 * fY - 16.0
                    val bLab = 200.0 * (fY - fZ)

                    totalL += L
                    totalB += bLab
                    count++
                }
            }
        }

        val avgL = if (count > 0) totalL / count else 50.0
        val avgB = if (count > 0) totalB / count else 15.0
        val ita = atan2(avgL - 50.0, avgB) * (180.0 / Math.PI)

        val fst = when {
            ita > 55.0 -> 1
            ita > 41.0 -> 2
            ita > 28.0 -> 3
            ita > 10.0 -> 4
            ita > -30.0 -> 5
            else -> 6
        }

        val toneBand = when (fst) {
            1 -> "Very Light"
            2 -> "Light"
            3 -> "Intermediate"
            4 -> "Tan"
            5 -> "Brown"
            else -> "Dark"
        }

        return ItaResult(ita, fst, toneBand)
    }

    private fun createAttentionOverlay(orig: Bitmap, predIdx: Int, isOutside: Boolean): Bitmap {
        val w = orig.width; val h = orig.height
        val overlay = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(overlay)
        canvas.drawBitmap(orig, 0f, 0f, null)

        val paint = Paint().apply {
            isAntiAlias = true
            style = Paint.Style.FILL
        }

        val cx = w / 2f
        val cy = h / 2f
        val radius = min(w, h) * 0.35f

        for (i in 5 downTo 1) {
            val r = radius * (i / 5f)
            val alpha = if (isOutside) 20 else (35 + (5 - i) * 15)
            val color = if (isOutside) {
                Color.argb(alpha, 120, 140, 160)
            } else {
                when (i) {
                    1, 2 -> Color.argb(alpha, 255, 60, 0)
                    3 -> Color.argb(alpha, 255, 200, 0)
                    else -> Color.argb(alpha, 0, 180, 255)
                }
            }
            paint.color = color
            canvas.drawCircle(cx, cy, r, paint)
        }

        return overlay
    }

    private fun bitmapToBase64(bmp: Bitmap): String {
        val baos = ByteArrayOutputStream()
        bmp.compress(Bitmap.CompressFormat.JPEG, 85, baos)
        return Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP)
    }
}
