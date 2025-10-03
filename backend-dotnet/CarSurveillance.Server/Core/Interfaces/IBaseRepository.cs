namespace CarSurveillance.Server.Core.Interfaces;

public interface IBaseRepository<T> where T: class, IBaseEntity
{
    Task<T?> GetAsync(Guid id, CancellationToken token);

    Task<List<T>> GetAsync(IEnumerable<Guid> ids, CancellationToken token);

    Task<List<T>> GetAllAsync(CancellationToken token);

    Task<bool> DeleteAsync(Guid id, CancellationToken token);

    Task<T> CreateAsync(T entity, CancellationToken token);

    Task<T> UpdateAsync(T entity, CancellationToken token);
}