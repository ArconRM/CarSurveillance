using System.Text.Json;
using CarSurveillance.Server.Dto.Requests;
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

    public async Task CropToLicensePlatesAsync(string rawDataPath, string resultDataPath, CancellationToken token)
    {
        var request = new CropToLicensePlatesRequest
        {
            RawDataPath = rawDataPath,
            ResultDataPath = resultDataPath
        };

        var response = await _httpClient.PostAsJsonAsync("api/cropToLicensePlates", request,
            _jsonSerializerOptions, token);
        await EnsureSuccessStatusCodeAsync(response, token);
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
}