import SwiftUI
import AVFoundation
import CoreML
import Vision
import UIKit

struct ContentView: View {
    let model: best
    @StateObject private var cameraManager = CameraManager()
    @State private var isRunning = false
    @State private var predictionText = "Waiting for Prediction..."

    init() {
        // Load the Core ML model
        do {
            model = try best(configuration: MLModelConfiguration())
        } catch {
            fatalError("Failed to load Core ML model: \(error)")
        }
    }

    var body: some View {
        ZStack {
            // Live Camera Feed
            CameraPreview(session: cameraManager.session)
                .edgesIgnoringSafeArea(.all)

            // UI Overlay
            VStack {
                Spacer()

                // Prediction Box
                Text(predictionText)
                    .font(.title)
                    .fontWeight(.bold)
                    .foregroundColor(.white)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color.black.opacity(0.7))
                    .cornerRadius(10)
                    .padding()

                // Start/Stop Buttons
                HStack {
                    Button("Start") {
                        isRunning = true
                        cameraManager.startProcessing { image in
                            if isRunning {
                                runModel(image: image)
                            }
                        }
                    }
                    .padding()
                    .background(Color.green)
                    .foregroundColor(.white)
                    .cornerRadius(10)

                    Button("Stop") {
                        isRunning = false
                        cameraManager.stopProcessing()
                    }
                    .padding()
                    .background(Color.red)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
                .padding()
            }
        }
        .onAppear {
            cameraManager.startSession()
        }
    }

    func runModel(image: UIImage) {
        guard let pixelBuffer = image.resized(to: CGSize(width: 640, height: 640))?.toPixelBuffer() else {
            print("❌ Error: Image processing failed")
            return
        }

        do {
            let prediction = try model.prediction(image: pixelBuffer, iouThreshold: 0.5, confidenceThreshold: 0.5)
            if let confidenceArray = prediction.featureValue(for: "confidence")?.multiArrayValue?.toArray() {
                let softmaxScores = softmax(confidenceArray)
                if softmaxScores.count >= 2 {
                    let predictedClass = softmaxScores[0] > softmaxScores[1] ? "Fake" : "Real"
                    DispatchQueue.main.async {
                        predictionText = "🔹 \(predictedClass)"
                    }
                } else {
                    DispatchQueue.main.async {
                        predictionText = "⚠️ No Prediction"
                    }
                }
            }
        } catch {
            print("❌ Error running YOLOv8: \(error.localizedDescription)")
        }
    }

    func softmax(_ values: [Float]) -> [Float] {
        let expValues = values.map { expf($0) }
        let sumExpValues = expValues.reduce(0, +)
        return expValues.map { $0 / sumExpValues }
    }
}

// Camera Preview (Live Feed)
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: UIScreen.main.bounds)
        let previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = view.bounds
        view.layer.addSublayer(previewLayer)
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {}
}

// Camera Manager (Handles Camera + Flash)
class CameraManager: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private var frameProcessingCallback: ((UIImage) -> Void)?
    private var isProcessing = false
    private var cameraDevice: AVCaptureDevice?

    override init() {
        super.init()
        setupCamera()
    }

    func setupCamera() {
        session.sessionPreset = .hd1920x1080  // High resolution

        guard let camera = getBestCamera() else {
            print("❌ Error: No back camera available")
            return
        }
        self.cameraDevice = camera

        do {
            let input = try AVCaptureDeviceInput(device: camera)
            session.addInput(input)
            videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "cameraFrameQueue"))
            session.addOutput(videoOutput)
        } catch {
            print("❌ Error: Unable to access camera: \(error.localizedDescription)")
        }
    }

    func startSession() {
        DispatchQueue.global(qos: .userInitiated).async {
            self.session.startRunning()
            print("📷 Camera session started")
            self.turnFlashOn()
        }
    }

    func startProcessing(callback: @escaping (UIImage) -> Void) {
        frameProcessingCallback = callback
        isProcessing = true
    }

    func stopProcessing() {
        isProcessing = false
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard isProcessing, let image = sampleBuffer.toUIImage() else { return }
        DispatchQueue.main.async {
            print("📸 Processing new frame")
            self.frameProcessingCallback?(image)
        }
    }

    func getBestCamera() -> AVCaptureDevice? {
        let discoverySession = AVCaptureDevice.DiscoverySession(
            deviceTypes: [
                .builtInUltraWideCamera,
                .builtInWideAngleCamera
            ],
            mediaType: .video,
            position: .back
        )
        return discoverySession.devices.first
    }

    func turnFlashOn() {
        guard let device = cameraDevice, device.hasTorch else {
            print("⚠️ Flash not available on this device")
            return
        }

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try device.lockForConfiguration()
                if device.isTorchModeSupported(.on) {
                    device.torchMode = .on
                    print("✅ Flash turned ON")
                } else {
                    print("⚠️ Torch mode not supported")
                }
                device.unlockForConfiguration()
            } catch {
                print("❌ Error enabling flash: \(error.localizedDescription)")
            }
        }
    }
}
