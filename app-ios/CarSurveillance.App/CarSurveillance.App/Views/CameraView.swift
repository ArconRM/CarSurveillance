//
//  CameraView.swift
//  CarSurveillance.App
//
//  Created by Artemiy MIROTVORTSEV on 28.09.2025.
//

import SwiftUI

struct CameraView: View {
    @EnvironmentObject private var frameViewModel: FrameViewModel
    @EnvironmentObject private var recordingViewModel: RecordingViewModel
    @State private var showBlackScreen: Bool = false
    @State private var showPreviewFullScreen: Bool = false
    
    var body: some View {
        ZStack {
            VStack {
                FrameView(image: $frameViewModel.frame)
                    .onTapGesture {
                        showPreviewFullScreen.toggle()
                    }
                if !showPreviewFullScreen {
                    VStack {
                        VStack(spacing: 8) {
                            HStack {
                                Circle()
                                    .fill(recordingViewModel.isRecording ? .red : .gray)
                                    .frame(width: 12, height: 12)
                                
                                Text(recordingViewModel.isRecording ? "RECORDING" : "STOPPED")
                                    .font(.headline)
                                    .foregroundColor(recordingViewModel.isRecording ? .red : .gray)
                                
                                Spacer()
                                
                                Text("Frames: \(recordingViewModel.framesCaptured)")
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                            }
                            
                            Text("Status: \(recordingViewModel.uploadStatus)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding(.horizontal)
                        .padding(.top)
                        
                        // Recording Control Button
                        Button(action: {
                            if recordingViewModel.isRecording {
                                recordingViewModel.stopRecording()
                            } else {
                                recordingViewModel.startRecording()
                            }
                        }) {
                            HStack {
                                Image(systemName: recordingViewModel.isRecording ? "stop.circle.fill" : "record.circle")
                                    .font(.title2)
                                Text(recordingViewModel.isRecording ? "Stop Recording" : "Start Recording")
                                    .font(.headline)
                            }
                            .foregroundColor(.white)
                            .padding()
                            .background(recordingViewModel.isRecording ? .red : .green)
                            .cornerRadius(10)
                        }
                        .padding(.horizontal)
                        
                        HStack(spacing: 20) {
                            // ISO Control
                            VStack(alignment: .leading) {
                                Text("ISO: \(Int(frameViewModel.currentISO))")
                                    .font(.headline)
                                Slider(value: Binding(
                                    get: { frameViewModel.currentISO },
                                    set: { frameViewModel.setISO($0) }
                                ), in: frameViewModel.minISO...frameViewModel.maxISO)
                            }
                            .padding(.horizontal)
                            
                            // Shutter Speed Control
                            VStack(alignment: .leading) {
                                Text("Shutter Speed: \(frameViewModel.formatShutterSpeed(frameViewModel.currentShutterSpeed))")
                                    .font(.headline)
                                Slider(value: Binding(
                                    get: { frameViewModel.currentShutterSpeed },
                                    set: { frameViewModel.setShutterSpeed($0) }
                                ), in: frameViewModel.minShutterSpeed...frameViewModel.maxShutterSpeed)
                            }
                            .padding(.horizontal)
                            
                            // Zoom control
                            VStack(alignment: .leading) {
                                Text("Zoom: \(frameViewModel.currentZoom)")
                                    .font(.headline)
                                Slider(value: Binding(
                                    get: { frameViewModel.currentZoom },
                                    set: { frameViewModel.setZoom($0) }
                                ), in: frameViewModel.minZoom...frameViewModel.maxZoom)
                            }
                            .padding(.horizontal)
                        }
                        .padding()
                        .background(.ultraThinMaterial)
                    }
                }
            }
            
            if showBlackScreen {
                Color.black
                    .ignoresSafeArea()
                    .onTapGesture {
                        showBlackScreen.toggle()
                    }
            }
        }
    }
}
