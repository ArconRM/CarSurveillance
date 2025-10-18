using CarSurveillance.Server.Core.BaseEntities;
using CarSurveillance.Server.Entities;
using CarSurveillance.Server.Repository.Interfaces;
using CarSurveillance.Server.Service.Interfaces;

namespace CarSurveillance.Server.Service;

public class CarPassService : BaseService<CarPassRecord>, ICarPassService
{
    public CarPassService(ICarPassRepository repository) : base(repository) { }
}