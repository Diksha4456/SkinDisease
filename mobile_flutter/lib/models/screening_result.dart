import 'dart:convert';
import 'dart:typed_data';

class ScreeningResult {
  final bool success;
  final String predictedDisease;
  final double confidencePct;
  final double entropy;
  final String uncertaintyLevel;
  final String acneSeverity;
  final int fitzpatrickScale;
  final double itaDegrees;
  final String itaToneBand;
  final String? qualityWarning;
  final List<String> observableFeatures;
  final String clinicalRationale;
  final String referralGuidance;
  final Uint8List? originalBytes;
  final Uint8List? gradcamBytes;

  ScreeningResult({
    required this.success,
    required this.predictedDisease,
    required this.confidencePct,
    required this.entropy,
    required this.uncertaintyLevel,
    required this.acneSeverity,
    required this.fitzpatrickScale,
    required this.itaDegrees,
    required this.itaToneBand,
    this.qualityWarning,
    required this.observableFeatures,
    required this.clinicalRationale,
    required this.referralGuidance,
    this.originalBytes,
    this.gradcamBytes,
  });

  factory ScreeningResult.fromJson(Map<String, dynamic> json) {
    Uint8List? orig;
    Uint8List? grad;

    if (json['original_b64'] != null && json['original_b64'].toString().isNotEmpty) {
      try {
        orig = base64Decode(json['original_b64']);
      } catch (_) {}
    }

    if (json['gradcam_b64'] != null && json['gradcam_b64'].toString().isNotEmpty) {
      try {
        grad = base64Decode(json['gradcam_b64']);
      } catch (_) {}
    }

    return ScreeningResult(
      success: json['success'] ?? false,
      predictedDisease: json['predicted_disease'] ?? 'Unknown',
      confidencePct: (json['confidence_pct'] as num?)?.toDouble() ?? 0.0,
      entropy: (json['entropy'] as num?)?.toDouble() ?? 0.0,
      uncertaintyLevel: json['uncertainty_level'] ?? 'Normal',
      acneSeverity: json['acne_severity'] ?? 'N/A',
      fitzpatrickScale: json['fitzpatrick_scale'] ?? 3,
      itaDegrees: (json['ita_degrees'] as num?)?.toDouble() ?? 30.0,
      itaToneBand: json['ita_tone_band'] ?? 'Intermediate',
      qualityWarning: json['quality_warning'],
      observableFeatures: List<String>.from(json['observable_features'] ?? []),
      clinicalRationale: json['clinical_rationale'] ?? '',
      referralGuidance: json['referral_guidance'] ?? '',
      originalBytes: orig,
      gradcamBytes: grad,
    );
  }
}
