namespace CarSurveillance.Server.Core.Interfaces;

public interface IBaseService<T> where T : class, IBaseEntity
{
    Task<T?> GetAsync(Guid uuid, CancellationToken token);

    Task<IEnumerable<T>> GetAsync(IEnumerable<Guid> uuids, CancellationToken token);

    Task<IEnumerable<T>> GetAllAsync(CancellationToken token);

    Task<bool> DeleteAsync(Guid uuid, CancellationToken token);

    Task<T> CreateAsync(T entity, CancellationToken token);

    Task<T> UpdateAsync(T entity, CancellationToken token);
}