using CarSurveillance.Server.Core.BaseEntities;
using CarSurveillance.Server.Core.Interfaces;
using CarSurveillance.Server.Entities;
using CarSurveillance.Server.Service.Interfaces;

namespace CarSurveillance.Server.Service;

public class CarPassService: BaseService<CarPass>, ICarPassService
{
    public CarPassService(IBaseRepository<CarPass> repository) : base(repository) { }
}