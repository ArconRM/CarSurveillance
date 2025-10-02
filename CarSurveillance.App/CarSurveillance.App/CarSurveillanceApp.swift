//
//  CarSurveillanceApp.swift
//  CarSurveillance.App
//
//  Created by Artemiy MIROTVORTSEV on 28.09.2025.
//

import SwiftUI

@main
struct CarSurveillanceApp: App {
    let viewModelFactory: ViewModelFactory = .init()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(viewModelFactory.frameViewModel)
                .environmentObject(viewModelFactory.recordingViewModel)
        }
    }
}
