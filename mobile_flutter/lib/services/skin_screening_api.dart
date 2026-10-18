import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import '../models/screening_result.dart';

class SkinScreeningApi {
  // http://127.0.0.1:5000 for USB adb reverse / Desktop
  // http://10.0.2.2:5000 for Android Emulator
  // http://<YOUR_PC_IP>:5000 for physical phone over Wi-Fi
  String baseUrl;

  SkinScreeningApi({this.baseUrl = "http://127.0.0.1:5000"});

  void setBaseUrl(String url) {
    if (url.endsWith('/')) {
      baseUrl = url.substring(0, url.length - 1);
    } else {
      baseUrl = url;
    }
  }

  /// Check server health and model readiness
  Future<Map<String, dynamic>> checkHealth() async {
    try {
      final res = await http.get(Uri.parse('$baseUrl/api/health')).timeout(const Duration(seconds: 5));
      if (res.statusCode == 200) {
        return jsonDecode(res.body);
      }
      throw Exception('Server health check failed (${res.statusCode}): ${res.body}');
    } catch (e) {
      throw Exception('Could not connect to FairSkin AI backend ($baseUrl): $e');
    }
  }

  /// Screen a skin image file via standard multipart/form-data
  Future<ScreeningResult> screenImageFile(File imageFile) async {
    final uri = Uri.parse('$baseUrl/api/screen');
    final req = http.MultipartRequest('POST', uri);
    req.files.add(await http.MultipartFile.fromPath('image', imageFile.path));

    final streamedRes = await req.send().timeout(const Duration(seconds: 40));
    final res = await http.Response.fromStream(streamedRes);

    if (res.statusCode == 200) {
      final data = jsonDecode(res.body);
      return ScreeningResult.fromJson(data);
    } else {
      throw Exception('Screening failed (${res.statusCode}): ${res.body}');
    }
  }

  /// Screen a skin image from Base64 JSON payload
  Future<ScreeningResult> screenImageBase64(String base64String) async {
    final uri = Uri.parse('$baseUrl/api/screen_base64');
    final res = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'image_base64': base64String}),
    ).timeout(const Duration(seconds: 40));

    if (res.statusCode == 200) {
      final data = jsonDecode(res.body);
      return ScreeningResult.fromJson(data);
    } else {
      throw Exception('Screening failed (${res.statusCode}): ${res.body}');
    }
  }

  /// Fetch live Data Warehouse (DWH) analytics from Kimball Data Marts
  Future<Map<String, dynamic>> fetchDwhAnalytics() async {
    final uri = Uri.parse('$baseUrl/api/dwh/analytics');
    final res = await http.get(uri).timeout(const Duration(seconds: 10));

    if (res.statusCode == 200) {
      return jsonDecode(res.body);
    } else {
      throw Exception('Failed to fetch DWH analytics: ${res.statusCode}');
    }
  }
}
