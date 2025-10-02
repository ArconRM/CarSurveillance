using System.IO.Compression;
using CarSurveillance.Server.Options;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;

namespace CarSurveillance.Server.Controllers;

[ApiController]
[Route("api/[controller]")]
public class DataController: ControllerBase
{
    private readonly string _dataPath;
    private readonly ILogger<DataController> _logger;

    public DataController(IOptions<DataOptions> options, ILogger<DataController> logger)
    {
        _dataPath = options.Value.DataPath;
        _logger = logger;
    }

    [HttpPost(nameof(UploadZip))]
    public async Task<IActionResult> UploadZip(IFormFile inputZip, CancellationToken token)
    {
        try
        {
            await using var inputStream = inputZip.OpenReadStream();
            
            if (Directory.Exists(_dataPath))
            {
                var files = Directory.GetFiles(_dataPath);
                var directories = Directory.GetDirectories(_dataPath);
            
                foreach (var file in files)
                {
                    System.IO.File.Delete(file);
                }
                
                foreach (var dir in directories)
                {
                    Directory.Delete(dir, true);
                }
            }
            else
            {
                Directory.CreateDirectory(_dataPath);
            }
            
            Directory.CreateDirectory(_dataPath);
            using var archive = new ZipArchive(inputStream, ZipArchiveMode.Read);
            
            foreach (var entry in archive.Entries)
            {
                if (string.IsNullOrEmpty(entry.Name))
                    continue;

                var extension = Path.GetExtension(entry.Name).ToLowerInvariant();
                if (extension is not (".jpg" or ".jpeg" or ".png" or ".gif" or ".bmp" or ".webp"))
                    continue;

                var destinationPath = Path.Combine(_dataPath, entry.Name);
                
                await using var entryStream = entry.Open();
                await using var fileStream = new FileStream(destinationPath, FileMode.Create, FileAccess.Write);
                await entryStream.CopyToAsync(fileStream, token);
            }
            
            return Ok();
        }
        catch (Exception e)
        {
            _logger.LogError(e, e.Message);
            throw;
        }
    }
    
    [HttpPost(nameof(UploadBatch))]
    public async Task<IActionResult> UploadBatch(List<IFormFile> images, CancellationToken token)
    {
        try
        {
            if (Directory.Exists(_dataPath))
            {
                foreach (var file in Directory.GetFiles(_dataPath))
                    System.IO.File.Delete(file);

                foreach (var dir in Directory.GetDirectories(_dataPath))
                    Directory.Delete(dir, true);
            }
            else
            {
                Directory.CreateDirectory(_dataPath);
            }

            foreach (var image in images)
            {
                var extension = Path.GetExtension(image.FileName).ToLowerInvariant();
                if (extension is not (".jpg" or ".jpeg" or ".png" or ".gif" or ".bmp" or ".webp"))
                    continue;

                var destinationPath = Path.Combine(_dataPath, image.FileName);

                await using var stream = new FileStream(destinationPath, FileMode.Create);
                await image.CopyToAsync(stream, token);
            }

            return Ok(new { Count = images.Count });
        }
        catch (Exception e)
        {
            _logger.LogError(e, e.Message);
            throw;
        }
    }
}