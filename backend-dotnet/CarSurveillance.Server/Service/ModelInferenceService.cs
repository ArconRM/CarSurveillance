using CarSurveillance.Server.HttpService.Interfaces;
using CarSurveillance.Server.Service.Interfaces;

namespace CarSurveillance.Server.Service;

public class ModelInferenceService : IModelInferenceService
{
    private readonly ICarPassService _carPassService;
    private readonly IModelInferenceHttpService _modelInferenceHttpService;

    public ModelInferenceService(
        IModelInferenceHttpService modelInferenceHttpService,
        ICarPassService carPassService)
    {
        _modelInferenceHttpService = modelInferenceHttpService;
        _carPassService = carPassService;
    }

    public async Task CropToLicensePlatesAsync(
        string rawDataPath,
        string resultDataPath,
        CancellationToken token)
    {
        await _modelInferenceHttpService.CropToLicensePlatesAsync(rawDataPath, resultDataPath, token);
    }

    public async Task RecognizeLicensePlatesAndSaveResults(
        string cropsDataPath,
        string resultDataPath,
        CancellationToken token)
    {
        var carPassRecords = await _modelInferenceHttpService.RecognizeLicensePlates(
            cropsDataPath,
            resultDataPath,
            token);

        foreach (var carPassRecord in carPassRecords) await _carPassService.CreateAsync(carPassRecord, token);
    }
}