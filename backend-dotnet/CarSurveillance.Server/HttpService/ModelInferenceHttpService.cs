using System.Text.Json;
using CarSurveillance.Server.Dto.Requests;
using CarSurveillance.Server.Dto.Responses;
using CarSurveillance.Server.Entities;
using CarSurveillance.Server.HttpService.Interfaces;

namespace CarSurveillance.Server.HttpService;

public class ModelInferenceHttpService : IModelInferenceHttpService
{
    private readonly HttpClient _httpClient;
    private readonly JsonSerializerOptions _jsonSerializerOptions = new() { PropertyNameCaseInsensitive = true };

    public ModelInferenceHttpService(HttpClient httpClient)
    {
        _httpClient = httpClient;
    }

    public async Task CropToLicensePlatesAsync(string rawDataPath, string cropsDataPath, CancellationToken token)
    {
        var request = new DataCropInferenceRequest
        {
            RawDataPath = rawDataPath,
            CropsDataPath = cropsDataPath,
        };

        var response = await _httpClient.PostAsJsonAsync("api/cropToLicensePlates", request,
            _jsonSerializerOptions, token);
        await EnsureSuccessStatusCodeAsync(response, token);
    }

    public async Task<IEnumerable<CarPassRecord>> RecognizeLicensePlates(
        string cropsDataPath, string resultDataPath, CancellationToken token)
    {
        var request = new DataRecognizeInferenceRequest
        {
            CropsDataPath = cropsDataPath,
            ResultDataPath = resultDataPath
        };

        var response = await _httpClient.PostAsJsonAsync("api/recognizeLicensePlates", request,
            _jsonSerializerOptions, token);
        await EnsureSuccessStatusCodeAsync(response, token);

        var json = await response.Content.ReadAsStringAsync(token);
        return DeserializeRecognitionData(json);
    }

    private async Task EnsureSuccessStatusCodeAsync(HttpResponseMessage response, CancellationToken token)
    {
        if (!response.IsSuccessStatusCode)
        {
            var errorContent = await response.Content.ReadAsStringAsync(token);
            throw new HttpRequestException(
                $"API call failed with status code {response.StatusCode}: {errorContent}",
                null,
                response.StatusCode
            );
        }
    }

    private IEnumerable<CarPassRecord> DeserializeRecognitionData(string jsonResponse)
    {
        var options = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        };

        var recognitionData = JsonSerializer.Deserialize<RecognitionResponse>(jsonResponse, options);

        return recognitionData.Results.Select(r => new CarPassRecord
        {
            DateTime = DateTimeOffset.FromUnixTimeSeconds(long.Parse(r.Time)).UtcDateTime,
            Filename = r.Filename,
            PlateTextRaw = r.PlateTextRaw,
            ConfidenceRaw = r.ConfidenceRaw,
            PlateTextProcessed = r.PlateTextProcessed,
            ConfidenceProcessed = r.ConfidenceProcessed
        });
    }
}