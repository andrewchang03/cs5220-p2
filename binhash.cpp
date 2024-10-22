#include <string.h>
#include <omp.h>
#include <vector>

#include "zmorton.hpp"
#include "binhash.hpp"

/*@q
 * ====================================================================
 */

/*@T
 * \subsection{Spatial hashing implementation}
 *
 * In the current implementation, we assume [[HASH_DIM]] is $2^b$,
 * so that computing a bitwise of an integer with [[HASH_DIM]] extracts
 * the $b$ lowest-order bits.  We could make [[HASH_DIM]] be something
 * other than a power of two, but we would then need to compute an integer
 * modulus or something of that sort.
 *
 *@c*/

#define HASH_MASK (HASH_DIM - 1)

// std::vector<omp_lock_t> locks;

// int init()
// {
//     locks = std::vector<omp_lock_t>(HASH_SIZE);
// #pragma omp parallel for
//     for (int i = 0; i < HASH_SIZE; i++)
//     {
//         omp_init_lock(&locks[i]);
//     }
//     return 0;
// }

// int d = init();

unsigned particle_bucket(particle_t *p, float h)
{
    unsigned ix = p->x[0] / h;
    unsigned iy = p->x[1] / h;
    unsigned iz = p->x[2] / h;
    return zm_encode(ix & HASH_MASK, iy & HASH_MASK, iz & HASH_MASK);
}

unsigned particle_neighborhood(unsigned *buckets, particle_t *p, float h)
{
    /* BEGIN TASK */
    // extract ix, iy, iz
    unsigned ix = p->x[0] / h;
    unsigned iy = p->x[1] / h;
    unsigned iz = p->x[2] / h;

    unsigned num_bins = 0;

    for (int dx = -1; dx <= 1; dx++)
    {
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dz = -1; dz <= 1; dz++)
            {
                if (dx == 0 && dy == 0 && dz == 0)
                    continue;

                unsigned jx = ix + dx;
                unsigned jy = iy + dy;
                unsigned jz = iz + dz;

                unsigned zm_index = zm_encode(jx & HASH_MASK, jy & HASH_MASK, jz & HASH_MASK);
                buckets[num_bins++] = zm_index;
            }
        }
    }

    return num_bins;
    /* END TASK */
}

void hash_particles(sim_state_t *s, float h)
{
    /* BEGIN TASK */
    memset(s->hash, 0, HASH_SIZE * sizeof(particle_t *));
#pragma omp parallel for
    for (int i = 0; i < s->n; i++)
    { // iterate through each particle
        particle_t *pi = &s->part[i];
        unsigned zm_index = particle_bucket(pi, h); // calculate zm index (the hash)
        // omp_set_lock(&locks[zm_index]);
        pi->next = s->hash[zm_index]; // prepend current particle to hash bucket
        s->hash[zm_index] = pi;       // update head
        // omp_unset_lock(&locks[zm_index]);
    }
    /* END TASK */
}
