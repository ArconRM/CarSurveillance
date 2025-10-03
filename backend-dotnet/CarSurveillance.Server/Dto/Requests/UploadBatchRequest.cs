namespace CarSurveillance.Server.Dto.Requests;

public class UploadBatchRequest
{
    public List<IFormFile> Images { get; set; }
}