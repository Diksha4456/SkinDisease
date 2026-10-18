import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../models/screening_result.dart';
import '../services/skin_screening_api.dart';

class ScreeningHomeScreen extends StatefulWidget {
  const ScreeningHomeScreen({super.key});

  @override
  State<ScreeningHomeScreen> createState() => _ScreeningHomeScreenState();
}

class _ScreeningHomeScreenState extends State<ScreeningHomeScreen> {
  final SkinScreeningApi _api = SkinScreeningApi();
  final ImagePicker _picker = ImagePicker();

  File? _selectedImage;
  ScreeningResult? _result;
  bool _isLoading = false;
  String? _errorMessage;

  Future<void> _pickImage(ImageSource source) async {
    try {
      final pickedFile = await _picker.pickImage(source: source, maxWidth: 1024, maxHeight: 1024);
      if (pickedFile != null) {
        setState(() {
          _selectedImage = File(pickedFile.path);
          _result = null;
          _errorMessage = null;
        });
        _runScreening();
      }
    } catch (e) {
      setState(() => _errorMessage = 'Failed to select image: $e');
    }
  }

  Future<void> _runScreening() async {
    if (_selectedImage == null) return;
    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    try {
      final res = await _api.screenImageFile(_selectedImage!);
      setState(() {
        _result = res;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _errorMessage = e.toString();
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0F172A),
      appBar: AppBar(
        title: const Text('FairSkin AI Screening', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18)),
        backgroundColor: const Color(0xFF1E293B),
        elevation: 0,
        actions: [
          IconButton(
            icon: const Icon(Icons.settings_ethernet),
            tooltip: 'Server Connection',
            onPressed: _showServerConfigDialog,
          ),
          IconButton(
            icon: const Icon(Icons.analytics_outlined),
            tooltip: 'DWH Analytics',
            onPressed: _showDwhDialog,
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Image Upload & Actions Card
            Card(
              color: const Color(0xFF1E293B),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  children: [
                    if (_selectedImage != null)
                      ClipRRect(
                        borderRadius: BorderRadius.circular(8),
                        child: Image.file(_selectedImage!, height: 220, width: double.infinity, fit: BoxFit.cover),
                      )
                    else
                      Container(
                        height: 180,
                        width: double.infinity,
                        decoration: BoxDecoration(
                          color: const Color(0xFF0F172A),
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(color: const Color(0xFF334155), width: 1.5),
                        ),
                        child: const Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.add_photo_alternate_outlined, size: 48, color: Color(0xFF0284C7)),
                            SizedBox(height: 8),
                            Text('Capture or Upload Skin Photo', style: TextStyle(color: Color(0xFF94A3B8), fontSize: 14)),
                          ],
                        ),
                      ),
                    const SizedBox(height: 16),
                    Row(
                      children: [
                        Expanded(
                          child: ElevatedButton.icon(
                            style: ElevatedButton.styleFrom(
                              backgroundColor: const Color(0xFF0284C7),
                              padding: const EdgeInsets.symmetric(vertical: 12),
                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                            ),
                            icon: const Icon(Icons.camera_alt, color: Colors.white),
                            label: const Text('Camera', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
                            onPressed: () => _pickImage(ImageSource.camera),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: ElevatedButton.icon(
                            style: ElevatedButton.styleFrom(
                              backgroundColor: const Color(0xFF334155),
                              padding: const EdgeInsets.symmetric(vertical: 12),
                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                            ),
                            icon: const Icon(Icons.photo_library, color: Colors.white),
                            label: const Text('Gallery', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
                            onPressed: () => _pickImage(ImageSource.gallery),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),

            if (_isLoading) ...[
              const SizedBox(height: 24),
              const Center(
                child: Column(
                  children: [
                    CircularProgressIndicator(color: Color(0xFF0284C7)),
                    SizedBox(height: 12),
                    Text('Running Multi-Task Deep Learning...', style: TextStyle(color: Color(0xFF94A3B8))),
                  ],
                ),
              ),
            ],

            if (_errorMessage != null) ...[
              const SizedBox(height: 16),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: const Color.fromRGBO(239, 68, 68, 0.15),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: const Color.fromRGBO(239, 68, 68, 0.5)),
                ),
                child: Text(_errorMessage!, style: const TextStyle(color: Colors.redAccent)),
              ),
            ],

            if (_result != null && !_isLoading) ...[
              const SizedBox(height: 16),
              _buildReportCard(_result!),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildReportCard(ScreeningResult res) {
    return Card(
      color: const Color(0xFF1E293B),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Expanded(
                  child: Text(
                    res.predictedDisease,
                    style: const TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: Color(0xFF38BDF8)),
                  ),
                ),
                if (res.acneSeverity != 'N/A')
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                    decoration: BoxDecoration(
                      color: const Color.fromRGBO(2, 132, 199, 0.25),
                      borderRadius: BorderRadius.circular(6),
                      border: Border.all(color: const Color(0xFF0284C7)),
                    ),
                    child: Text('${res.acneSeverity} Severity', style: const TextStyle(color: Color(0xFF38BDF8), fontSize: 12)),
                  ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text('Softmax Confidence', style: TextStyle(color: Color(0xFF94A3B8), fontSize: 13)),
                Text('${res.confidencePct.toStringAsFixed(1)}%', style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
              ],
            ),
            const SizedBox(height: 6),
            ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: res.confidencePct / 100.0,
                backgroundColor: const Color(0xFF334155),
                color: const Color(0xFF38BDF8),
                minHeight: 8,
              ),
            ),
            const SizedBox(height: 16),
            _buildMetaRow('Skin Tone (ITA)', '${res.itaToneBand} (FST ${res.fitzpatrickScale}) / ${res.itaDegrees.toStringAsFixed(1)}°'),
            _buildMetaRow('Uncertainty (Entropy)', '${res.entropy.toStringAsFixed(3)} (${res.uncertaintyLevel})'),

            const SizedBox(height: 16),
            const Text('Identified Visual Signs:', style: TextStyle(color: Color(0xFF94A3B8), fontWeight: FontWeight.bold, fontSize: 13)),
            const SizedBox(height: 8),
            Wrap(
              spacing: 6,
              runSpacing: 6,
              children: res.observableFeatures.map((f) => Chip(
                label: Text(f, style: const TextStyle(fontSize: 11, color: Color(0xFF38BDF8))),
                backgroundColor: const Color(0xFF0F172A),
                side: const BorderSide(color: Color(0xFF334155)),
                padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 0),
              )).toList(),
            ),

            if (res.originalBytes != null && res.gradcamBytes != null) ...[
              const SizedBox(height: 16),
              const Text('Explainable AI (Grad-CAM Salience):', style: TextStyle(color: Color(0xFF94A3B8), fontWeight: FontWeight.bold, fontSize: 13)),
              const SizedBox(height: 8),
              Row(
                children: [
                  Expanded(
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: Image.memory(res.originalBytes!, height: 130, fit: BoxFit.cover),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: Image.memory(res.gradcamBytes!, height: 130, fit: BoxFit.cover),
                    ),
                  ),
                ],
              ),
            ],

            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: const Color(0xFF0F172A),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Clinical Rationale:', style: TextStyle(fontWeight: FontWeight.bold, color: Colors.white, fontSize: 12)),
                  const SizedBox(height: 4),
                  Text(res.clinicalRationale, style: const TextStyle(color: Color(0xFFCBD5E1), fontSize: 12, height: 1.4)),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMetaRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(color: Color(0xFF94A3B8), fontSize: 13)),
          Text(value, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600, fontSize: 13)),
        ],
      ),
    );
  }

  void _showDwhDialog() async {
    showDialog(
      context: context,
      builder: (_) => const Center(child: CircularProgressIndicator(color: Color(0xFF0284C7))),
    );

    try {
      final dwh = await _api.fetchDwhAnalytics();
      if (!mounted) return;
      Navigator.pop(context); // close loader

      showDialog(
        context: context,
        builder: (ctx) => AlertDialog(
          backgroundColor: const Color(0xFF1E293B),
          title: const Text('Data Warehouse (DWH) Marts', style: TextStyle(color: Colors.white)),
          content: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text('Total Screenings: ${dwh['total_screenings']}', style: const TextStyle(color: Color(0xFF38BDF8), fontWeight: FontWeight.bold)),
                const Divider(color: Color(0xFF334155)),
                const Text('Fairness by Fitzpatrick Tone (FST I-VI):', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 13)),
                const SizedBox(height: 6),
                ...?((dwh['fairness_audit'] as List?)?.map((f) => Padding(
                  padding: const EdgeInsets.symmetric(vertical: 2.0),
                  child: Text('FST ${f['fst']} (${f['tone']}): count=${f['count']} | avg=${f['avg_confidence']}%',
                    style: const TextStyle(color: Color(0xFF94A3B8), fontSize: 12),
                  ),
                ))),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx),
              child: const Text('Close', style: TextStyle(color: Color(0xFF38BDF8))),
            ),
          ],
        ),
      );
    } catch (e) {
      if (!mounted) return;
      Navigator.pop(context);
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('DWH Error: $e')));
    }
  }

  void _showServerConfigDialog() {
    final controller = TextEditingController(text: _api.baseUrl);
    String statusMsg = '';
    Color statusColor = Colors.white;

    showDialog(
      context: context,
      builder: (ctx) => StatefulBuilder(
        builder: (context, setDialogState) => AlertDialog(
          backgroundColor: const Color(0xFF1E293B),
          title: const Text('Backend Server Connection', style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold)),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('Enter FairSkin AI Backend URL:', style: TextStyle(color: Color(0xFF94A3B8), fontSize: 12)),
              const SizedBox(height: 8),
              TextField(
                controller: controller,
                style: const TextStyle(color: Colors.white, fontSize: 14),
                decoration: InputDecoration(
                  filled: true,
                  fillColor: const Color(0xFF0F172A),
                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                  hintText: 'http://127.0.0.1:5000',
                  hintStyle: const TextStyle(color: Color(0xFF64748B)),
                ),
              ),
              const SizedBox(height: 10),
              Wrap(
                spacing: 6,
                children: [
                  ActionChip(
                    label: const Text('USB (127.0.0.1:5000)', style: TextStyle(fontSize: 10, color: Color(0xFF38BDF8))),
                    backgroundColor: const Color(0xFF0F172A),
                    onPressed: () => setDialogState(() => controller.text = 'http://127.0.0.1:5000'),
                  ),
                  ActionChip(
                    label: const Text('Emulator (10.0.2.2:5000)', style: TextStyle(fontSize: 10, color: Color(0xFF38BDF8))),
                    backgroundColor: const Color(0xFF0F172A),
                    onPressed: () => setDialogState(() => controller.text = 'http://10.0.2.2:5000'),
                  ),
                ],
              ),
              if (statusMsg.isNotEmpty) ...[
                const SizedBox(height: 10),
                Text(statusMsg, style: TextStyle(color: statusColor, fontSize: 12)),
              ],
            ],
          ),
          actions: [
            TextButton(
              onPressed: () async {
                setDialogState(() {
                  statusMsg = 'Testing connection...';
                  statusColor = Colors.amber;
                });
                _api.setBaseUrl(controller.text.trim());
                try {
                  final h = await _api.checkHealth();
                  setDialogState(() {
                    statusMsg = 'Connected: Status=${h['status']}';
                    statusColor = Colors.greenAccent;
                  });
                } catch (e) {
                  setDialogState(() {
                    statusMsg = 'Error: $e';
                    statusColor = Colors.redAccent;
                  });
                }
              },
              child: const Text('Test Connection', style: TextStyle(color: Color(0xFF38BDF8))),
            ),
            ElevatedButton(
              style: ElevatedButton.styleFrom(backgroundColor: const Color(0xFF0284C7)),
              onPressed: () {
                _api.setBaseUrl(controller.text.trim());
                Navigator.pop(ctx);
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text('Backend URL set to: ${_api.baseUrl}')),
                );
              },
              child: const Text('Save', style: TextStyle(color: Colors.white)),
            ),
          ],
        ),
      ),
    );
  }
}
