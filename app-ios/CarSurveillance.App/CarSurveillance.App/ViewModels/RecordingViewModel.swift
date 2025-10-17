//
//  RecordingViewModel.swift
//  CarSurveillance.App
//
//  Created by Artemiy MIROTVORTSEV on 28.09.2025.
//

import Foundation
internal import Combine
import UIKit

class RecordingViewModel: ObservableObject {
    @Published var isRecording: Bool = false
    @Published var framesCaptured: Int = 0
    @Published var uploadStatus: String = "Ready"
    
    private weak var frameViewModel: FrameViewModel?
    
    private var frameTimer: Timer?
    private var uploadTimer: Timer?
    private var frameBuffer: [Data] = []
    
    private let framesPerSecond: Double = 1.0
    private let uploadIntervalMinutes: Double = 1.0
    private let apiUploadEndpoint = "http://192.168.1.46:5035/api/Data/UploadBatch"
    private let apiCanSendEndpoint = "http://192.168.1.46:5035/api/Data/CanSend"
    
    init(frameViewModel: FrameViewModel) {
        self.frameViewModel = frameViewModel
    }
    
    func startRecording() {
        guard !isRecording else { return }
        
        DispatchQueue.main.async {
            self.isRecording = true
            self.framesCaptured = 0
            self.uploadStatus = "Recording..."
        }
        
        framesCaptured = 0
        
        frameTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / framesPerSecond, repeats: true) { [weak self] _ in
            self?.captureFrame()
        }
        
        uploadTimer = Timer.scheduledTimer(withTimeInterval: uploadIntervalMinutes * 60, repeats: true) { [weak self] _ in
            self?.makeBatchAndUpload()
        }
        
        print("Started recording with \(framesPerSecond) FPS")
    }
    
    func stopRecording() {
        guard isRecording else { return }
        
        frameTimer?.invalidate()
        uploadTimer?.invalidate()
        frameTimer = nil
        uploadTimer = nil
        
        DispatchQueue.main.async {
            self.isRecording = false
            self.uploadStatus = "Stopped"
        }
        
        print("Stopped recording")
    }
    
    private func captureFrame() {
        guard let cgImage = frameViewModel?.frame else { return }
        
        let uiImage = UIImage(cgImage: cgImage)
        guard let jpegData = uiImage.jpegData(compressionQuality: 1.0) else {
            print("Failed to convert image to JPEG")
            return
        }
        
        frameBuffer.append(jpegData)
        framesCaptured += 1
    }
    
    private func makeBatchAndUpload() {
        var request = URLRequest(url: URL(string: apiCanSendEndpoint)!,timeoutInterval: Double.infinity)
        request.addValue("*/*", forHTTPHeaderField: "accept")
        
        request.httpMethod = "GET"
        
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            guard let data = data else {
                print(String(describing: error))
                return
            }
            
            guard let canSend = try? JSONDecoder().decode(Bool.self, from: data), canSend
            else {
                print("Not sending time")
                return
            }
            
            DispatchQueue.main.async {
                self.uploadStatus = "Zipping frames..."
            }
            
            DispatchQueue.global(qos: .utility).async { [weak self] in
                guard let self = self else { return }
                
                let timestamp = Int(Date().timeIntervalSince1970)
                let batch = frameBuffer.enumerated().map { index, data in
                    (name: "frame_\(timestamp)-\(index).jpg", data: data)
                }
                
                DispatchQueue.main.async {
                    self.uploadStatus = "Uploading..."
                }
                
                self.uploadImageBatch(batch) { [weak self] success in
                    DispatchQueue.main.async {
                        if success {
                            self?.uploadStatus = "Upload successful"
                        } else {
                            self?.uploadStatus = "Upload failed"
                        }
                        self?.cleanupFrameFiles()
                    }
                }
            }
        }
        
        task.resume()
    }
    
    private func uploadImageBatch(_ images: [(name: String, data: Data)], completion: @escaping (Bool) -> Void) {
        let boundary = "Boundary-\(UUID().uuidString)"
        var request = URLRequest(url: URL(string: apiUploadEndpoint)!)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        
        for image in images {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"images\"; filename=\"\(image.name)\"\r\n".data(using: .utf8)!)
            body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
            body.append(image.data)
            body.append("\r\n".data(using: .utf8)!)
        }
        
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = body
        
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                print("Upload error: \(error)")
                completion(false)
                return
            }
            guard let data = data else {
                completion(false)
                return
            }
            print(response ?? "No response")
            print("Server response:", String(data: data, encoding: .utf8) ?? "<no response>")
            completion(true)
        }
        
        task.resume()
    }
    
    
    private func cleanupFrameFiles() {
        frameBuffer.removeAll()
    }
}
