//
//  FrameView.swift
//  CarSurveillance.App
//
//  Created by Artemiy MIROTVORTSEV on 28.09.2025.
//

import SwiftUI

struct FrameView: View {
    @Binding var image: CGImage?
    private let label: Text = Text("frame")
    
    var body: some View {
        if image != nil {
            Image(image!, scale: 1.0, orientation: .up, label: label)
                .resizable()
                .scaledToFit()
        } else {
            Color.gray
                .scaledToFit()
        }
    }
}

#Preview {
    FrameView(image: .constant(nil))
}
