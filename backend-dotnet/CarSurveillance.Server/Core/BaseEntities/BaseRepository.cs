using CarSurveillance.Server.Core.Interfaces;
using Microsoft.EntityFrameworkCore;

namespace CarSurveillance.Server.Core.BaseEntities;

public abstract class BaseRepository<T>: IBaseRepository<T> where T: class, IBaseEntity
{
    private readonly DbContext _context;
    private readonly DbSet<T> _dbSet;

    protected BaseRepository(DbContext context)
    {
        _context = context;
        _dbSet = context.Set<T>();
    }

    public virtual async Task<T?> GetAsync(Guid uuid, CancellationToken token)
    {
        return await _dbSet
            .AsNoTracking()
            .FirstOrDefaultAsync(e => e.Uuid == uuid, token);
    }

    public virtual async Task<List<T>> GetAsync(IEnumerable<Guid> uuids, CancellationToken token)
    {
        return await _dbSet
            .AsNoTracking()
            .Where(e => uuids.Contains(e.Uuid))
            .ToListAsync(token);
    }

    public virtual async Task<List<T>> GetAllAsync(CancellationToken token)
    {
        return await _dbSet
            .AsNoTracking()
            .ToListAsync(token);
    }

    public virtual async Task<bool> DeleteAsync(Guid uuid, CancellationToken token)
    {
        var entity = await _dbSet.FindAsync([uuid], token);
            
        if (entity == null)
            return false;

        _dbSet.Remove(entity);
        await _context.SaveChangesAsync(token);
            
        return true;
    }

    public virtual async Task<T> CreateAsync(T entity, CancellationToken cancellationToken = default)
    {       
        await _dbSet.AddAsync(entity, cancellationToken);
        await _context.SaveChangesAsync(cancellationToken);
            
        return entity;
    }

    public virtual async Task<T> UpdateAsync(T entity, CancellationToken token)
    {
        _dbSet.Update(entity);
        await _context.SaveChangesAsync(token);
            
        return entity;
    }
}