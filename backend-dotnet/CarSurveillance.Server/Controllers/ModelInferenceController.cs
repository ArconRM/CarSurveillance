using CarSurveillance.Server.Dto.Requests;
using CarSurveillance.Server.HttpService.Interfaces;
using Microsoft.AspNetCore.Mvc;

namespace CarSurveillance.Server.Controllers;

[ApiController]
[Route("api/[controller]")]
public class ModelInferenceController : ControllerBase
{
    private readonly IModelInferenceHttpService _modelInferenceHttpService;

    public ModelInferenceController(IModelInferenceHttpService modelInferenceHttpService)
    {
        _modelInferenceHttpService = modelInferenceHttpService;
    }

    [HttpPost(nameof(RunDetectionInference))]
    public async Task<IActionResult> RunDetectionInference(DataInferenceRequest request, CancellationToken token)
    {
        await _modelInferenceHttpService.CropToLicensePlatesAsync(request.CropsDataPath, request.ResultDataPath, token);
        return Ok();
    }

    [HttpPost(nameof(RecognizeLicensePlates))]
    public async Task<IActionResult> RecognizeLicensePlates(DataInferenceRequest request, CancellationToken token)
    {
        await _modelInferenceHttpService.RecognizeLicensePlates(request.CropsDataPath, request.ResultDataPath, token);
        return Ok();
    }
}