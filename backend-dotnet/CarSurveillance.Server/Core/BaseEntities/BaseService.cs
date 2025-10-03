using CarSurveillance.Server.Core.Interfaces;

namespace CarSurveillance.Server.Core.BaseEntities;

public abstract class BaseService<T>: IBaseService<T> where T: class, IBaseEntity
{
    private readonly IBaseRepository<T> _repository;

    public BaseService(IBaseRepository<T> repository)
    {
        _repository = repository;
    }

    public async Task<T?> GetAsync(Guid uuid, CancellationToken token)
    {
        return await _repository.GetAsync(uuid, token); 
    }
    
    public async Task<IEnumerable<T>> GetAsync(IEnumerable<Guid> uuids, CancellationToken token)
    {
        return await _repository.GetAsync(uuids, token);
    }
    
    public async Task<IEnumerable<T>> GetAllAsync(CancellationToken token)
    {
        return await _repository.GetAllAsync(token);
    }
    
    public async Task<bool> DeleteAsync(Guid uuid, CancellationToken token)
    {
        return await _repository.DeleteAsync(uuid, token);
    }
    
    public async Task<T> CreateAsync(T entity, CancellationToken token)
    {
        return await _repository.CreateAsync(entity, token);
    }
    
    public async Task<T> UpdateAsync(T entity, CancellationToken token)
    {
        return await _repository.UpdateAsync(entity, token);
    }
}