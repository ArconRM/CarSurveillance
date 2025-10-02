//
//  FrameViewModel.swift
//  CarSurveillance.App
//
//  Created by Artemiy MIROTVORTSEV on 28.09.2025.
//

import AVFoundation
import CoreImage
internal import Combine
import UIKit

class FrameViewModel: NSObject, ObservableObject {
    @Published var frame: CGImage?
    @Published var currentISO: Float = 222
    @Published var currentShutterSpeed: Float = 1/1017.0
    @Published var currentZoom: CGFloat = 2.0
    @Published var minISO: Float = 100.0
    @Published var maxISO: Float = 200.0
    @Published var minShutterSpeed: Float = 1/100.0
    let maxShutterSpeed: Float = 1/60
    @Published var minZoom: CGFloat = 1.0
    @Published var maxZoom: CGFloat = 5.0
    
    private var permissionGranted = true
    private let captureSession = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "sessionQueue")
    private let context = CIContext()
    private var captureDevice: AVCaptureDevice?
    
    override init() {
        super.init()
        self.checkPermission()
        sessionQueue.async { [unowned self] in
            self.setupCaptureSession()
            self.captureSession.startRunning()
        }
    }
    
    func checkPermission() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            self.permissionGranted = true
        case .notDetermined:
            self.requestPermission()
        default:
            self.permissionGranted = false
        }
    }
    
    func requestPermission() {
        AVCaptureDevice.requestAccess(for: .video) { [unowned self] granted in
            self.permissionGranted = granted
        }
    }
    
    func setupCaptureSession() {
        let videoOutput = AVCaptureVideoDataOutput()
        
        guard permissionGranted else { return }
        guard let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else { return }
        
        self.captureDevice = videoDevice
        
        guard let videoDeviceInput = try? AVCaptureDeviceInput(device: videoDevice) else { return }
        guard captureSession.canAddInput(videoDeviceInput) else { return }
        captureSession.addInput(videoDeviceInput)
        
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "sampleBufferQueue"))
        captureSession.addOutput(videoOutput)
        
        videoOutput.connection(with: .video)?.videoOrientation = .landscapeRight
        
        // Configure camera for manual controls AFTER session is set up
        configureCameraForManualControls()
    }
    
    private func configureCameraForManualControls() {
        guard let device = captureDevice else {
            print("No capture device available")
            return
        }
        
        do {
            try device.lockForConfiguration()
            
            // Read device capabilities first
            let format = device.activeFormat
            let minISOValue = format.minISO
            let maxISOValue = format.maxISO
            
            // Get exposure duration limits as CMTime, then convert to Float
            let minExposureDuration = format.minExposureDuration
            let maxExposureDuration = format.maxExposureDuration
            
            let minShutterSpeedValue = Float(minExposureDuration.seconds)
            let maxShutterSpeedValue = Float(maxExposureDuration.seconds)
            
            let minZoomValue = device.minAvailableVideoZoomFactor
            let maxZoomValue = device.maxAvailableVideoZoomFactor
            
            print("Device capabilities:")
            print("ISO range: \(minISOValue) - \(maxISOValue)")
            print("Min exposure duration: \(minExposureDuration) (\(minShutterSpeedValue) seconds)")
            print("Max exposure duration: \(maxExposureDuration) (\(maxShutterSpeedValue) seconds)")
            print("Zoom range: \(minZoomValue)x - \(maxZoomValue)x")
            
            if device.isExposureModeSupported(.custom) {
                device.exposureMode = .custom
                
                // Calculate safe initial values within the device's supported range
                let initialISO = min(max(222, minISOValue), maxISOValue)
                
                // For shutter speed, ensure we're within the device's actual limits
                let desiredShutterSpeed: Float = 1.0/1017.0  // 1/60th second
                let initialShutterSpeed: Float
                
                if desiredShutterSpeed < minShutterSpeedValue {
                    // If 1/1017s is too fast, use the minimum (slowest) allowed
                    initialShutterSpeed = minShutterSpeedValue
                    print("1/1017s is too fast for this device, using minimum: \(minShutterSpeedValue)s")
                } else if desiredShutterSpeed > maxShutterSpeedValue {
                    // If 1/1017s is too slow, use the maximum (fastest) allowed
                    initialShutterSpeed = maxShutterSpeedValue
                    print("1/1017s is too slow for this device, using maximum: \(maxShutterSpeedValue)s")
                } else {
                    initialShutterSpeed = desiredShutterSpeed
                    print("Using desired 1/60s shutter speed")
                }
                
                print("Calculated initial values:")
                print("Initial ISO: \(initialISO)")
                print("Initial shutter speed: \(initialShutterSpeed)s")
                print("Initial zoom: \(minZoomValue)x")
                
                DispatchQueue.main.async { [weak self] in
                    self?.minISO = minISOValue
                    self?.maxISO = maxISOValue
                    self?.currentISO = initialISO
                    
                    self?.minShutterSpeed = minShutterSpeedValue
                    self?.currentShutterSpeed = initialShutterSpeed
                    
                    self?.minZoom = minZoomValue
                    self?.maxZoom = maxZoomValue
                    self?.currentZoom = minZoomValue
                }
                
                let initialDuration = CMTimeMakeWithSeconds(Double(initialShutterSpeed), preferredTimescale: minExposureDuration.timescale)
                
                print("Setting exposure with duration: \(initialDuration) and ISO: \(initialISO)")
                
                // Validate the duration is within bounds before setting
                if CMTimeCompare(initialDuration, minExposureDuration) >= 0 &&
                    CMTimeCompare(initialDuration, maxExposureDuration) <= 0 {
                    device.setExposureModeCustom(duration: initialDuration, iso: initialISO, completionHandler: nil)
                    print("Successfully set initial exposure settings")
                } else {
                    print("ERROR: Calculated duration \(initialDuration) is still outside bounds!")
                    print("Min: \(minExposureDuration), Max: \(maxExposureDuration)")
                    
                    // Fallback: use the current device exposure duration
                    device.setExposureModeCustom(duration: device.exposureDuration, iso: initialISO, completionHandler: nil)
                }
                
            } else {
                print("Custom exposure mode not supported")
            }
            
            device.unlockForConfiguration()
        } catch {
            print("Error configuring camera: \(error)")
        }
    }
    
    func setISO(_ iso: Float) {
        guard let device = captureDevice else { return }
        
        do {
            try device.lockForConfiguration()
            
            if device.isExposureModeSupported(.custom) {
                let clampedISO = min(max(iso, minISO), maxISO)
                device.setExposureModeCustom(duration: device.exposureDuration,
                                             iso: clampedISO) { _ in
                    DispatchQueue.main.async { [weak self] in
                        self?.currentISO = clampedISO
                    }
                }
                //                print("Set ISO to: \(clampedISO)")
            }
            
            device.unlockForConfiguration()
        } catch {
            print("Error setting ISO: \(error)")
        }
    }
    
    func setShutterSpeed(_ seconds: Float) {
        guard let device = captureDevice else { return }
        
        do {
            try device.lockForConfiguration()
            
            if device.isExposureModeSupported(.custom) {
                let clampedShutterSpeed = min(max(seconds, minShutterSpeed), maxShutterSpeed)
                
                let format = device.activeFormat
                let duration = CMTimeMakeWithSeconds(Double(clampedShutterSpeed), preferredTimescale: format.minExposureDuration.timescale)
                
                if CMTimeCompare(duration, format.minExposureDuration) >= 0 &&
                    CMTimeCompare(duration, format.maxExposureDuration) <= 0 {
                    
                    device.setExposureModeCustom(duration: duration, iso: device.iso) { _ in
                        DispatchQueue.main.async { [weak self] in
                            self?.currentShutterSpeed = clampedShutterSpeed
                        }
                    }
                    //                    print("Set shutter speed to: \(clampedShutterSpeed) seconds (duration: \(duration))")
                } else {
                    print("ERROR: Requested duration \(duration) is outside device bounds")
                    print("Min: \(format.minExposureDuration), Max: \(format.maxExposureDuration)")
                }
            }
            
            device.unlockForConfiguration()
        } catch {
            print("Error setting shutter speed: \(error)")
        }
    }
    
    func formatShutterSpeed(_ seconds: Float) -> String {
        if seconds >= 1.0 {
            return String(format: "%.0f\"", seconds)
        } else {
            return String(format: "1/%.0f", 1.0/seconds)
        }
    }
    
    func setZoom(_ zoomFactor: CGFloat) {
        guard let device = captureDevice else { return }
        
        do {
            try device.lockForConfiguration()
            
            // Clamp zoom factor to device limits
            let clampedZoom = min(max(zoomFactor, minZoom), maxZoom)
            device.videoZoomFactor = clampedZoom
            
            DispatchQueue.main.async { [weak self] in
                self?.currentZoom = clampedZoom
            }
            
            //            print("Set zoom to: \(clampedZoom)x")
            device.unlockForConfiguration()
        } catch {
            print("Error setting zoom: \(error)")
        }
    }
}

extension FrameViewModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let cgImage = imageFromSampleBuffer(sampleBuffer: sampleBuffer) else { return }
        
        DispatchQueue.main.async { [unowned self] in
            self.frame = cgImage
        }
    }
    
    private func imageFromSampleBuffer(sampleBuffer: CMSampleBuffer) -> CGImage? {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return nil }
        let ciImage = CIImage(cvPixelBuffer: imageBuffer)
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return nil }
        
        return cgImage
    }
}
