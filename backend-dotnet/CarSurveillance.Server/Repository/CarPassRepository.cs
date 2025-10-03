using CarSurveillance.Server.Core.BaseEntities;
using CarSurveillance.Server.Entities;
using CarSurveillance.Server.Repository.Interfaces;
using Microsoft.EntityFrameworkCore;

namespace CarSurveillance.Server.Repository;

public class CarPassRepository: BaseRepository<CarPass>, ICarPassRepository
{
    public CarPassRepository(DbContext context) : base(context) { }
}