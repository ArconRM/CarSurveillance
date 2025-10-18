using CarSurveillance.Server.Entities;

namespace CarSurveillance.Server.HttpService.Interfaces;

public interface IModelInferenceHttpService
{
    Task CropToLicensePlatesAsync(string rawDataPath, string cropsDataPath, CancellationToken token);

    Task<IEnumerable<CarPassRecord>> RecognizeLicensePlates(
        string cropsDataPath, string resultDataPath, CancellationToken token);
}