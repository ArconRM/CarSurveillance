//
//  ViewModelFactory.swift
//  CarSurveillance.App
//
//  Created by Artemiy MIROTVORTSEV on 29.09.2025.
//

import Foundation
import SwiftUI
internal import Combine

class ViewModelFactory: ObservableObject {
    @Published var frameViewModel: FrameViewModel
    @Published var recordingViewModel: RecordingViewModel
    
    init() {
        let _frameViewModel = FrameViewModel()
        self.frameViewModel = _frameViewModel
        self.recordingViewModel = RecordingViewModel(frameViewModel: _frameViewModel)
    }
}
