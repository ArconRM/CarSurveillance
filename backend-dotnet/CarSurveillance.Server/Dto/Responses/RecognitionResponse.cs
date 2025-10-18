using System.Text.Json.Serialization;

namespace CarSurveillance.Server.Dto.Responses;

public class RecognitionResponse
{
    [JsonPropertyName("status")] public string Status { get; set; }

    [JsonPropertyName("total_processed")] public int TotalProcessed { get; set; }

    [JsonPropertyName("results")] public List<RecognitionResult> Results { get; set; }
}

public class RecognitionResult
{
    [JsonPropertyName("time")] public string Time { get; set; }

    [JsonPropertyName("filename")] public string Filename { get; set; }

    [JsonPropertyName("plate_text_raw")] public string PlateTextRaw { get; set; }

    [JsonPropertyName("confidence_raw")] public double ConfidenceRaw { get; set; }

    [JsonPropertyName("plate_text_processed")]
    public string PlateTextProcessed { get; set; }

    [JsonPropertyName("confidence_processed")]
    public double ConfidenceProcessed { get; set; }
}