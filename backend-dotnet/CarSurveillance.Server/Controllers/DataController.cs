using CarSurveillance.Server.Dto.Requests;
using CarSurveillance.Server.Entities;
using CarSurveillance.Server.Options;
using CarSurveillance.Server.Service.Interfaces;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;

namespace CarSurveillance.Server.Controllers;

[ApiController]
[Route("api/[controller]")]
public class DataController : ControllerBase
{
    private readonly IFilesService _filesService;
    private readonly ILogger<DataController> _logger;
    private readonly DataProcessingOptions _options;

    public DataController(
        IOptions<DataProcessingOptions> options,
        IFilesService filesService,
        ILogger<DataController> logger)
    {
        _options = options.Value;
        _filesService = filesService;
        _logger = logger;
    }

    [HttpGet(nameof(CanSend))]
    public async Task<IActionResult> CanSend()
    {
        return Ok(DateTime.Now.Hour > _options.UploadingHourStart && DateTime.Now.Hour < _options.UploadingHourEnd);
    }

    [HttpPost(nameof(UploadZip))]
    public async Task<IActionResult> UploadZip(UploadZipRequest request, CancellationToken token)
    {
        await using var inputStream = request.InputZip.OpenReadStream();

        await _filesService.UploadZipAsync(inputStream, token);

        return Ok();
    }

    [HttpPost(nameof(UploadBatch))]
    public async Task<IActionResult> UploadBatch(UploadBatchRequest request, CancellationToken token)
    {
        var dtos = new List<UploadFile>();

        foreach (var file in request.Images)
            dtos.Add(new UploadFile
            {
                FileName = file.FileName,
                ContentType = file.ContentType,
                Content = file.OpenReadStream()
            });

        await _filesService.UploadBatchAsync(dtos, token);

        return Ok();
    }

    [HttpPost(nameof(CleanRawDataFolder))]
    public async Task<IActionResult> CleanRawDataFolder()
    {
        _filesService.CleanRawDataFolder();
        return Ok();
    }
}