#include <string.h>

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

#define HASH_MASK (HASH_DIM-1)

unsigned particle_bucket(particle_t* p, float h)
{
    unsigned ix = p->x[0]/h;
    unsigned iy = p->x[1]/h;
    unsigned iz = p->x[2]/h;
    return zm_encode(ix & HASH_MASK, iy & HASH_MASK, iz & HASH_MASK);
}

unsigned particle_neighborhood(unsigned* buckets, particle_t* p, float h)
{
    /* BEGIN TASK */
    // boundaries
    const unsigned XMAX = 1.0F / h;
    const unsigned YMAX = 1.0F / h;
    const unsigned ZMAX = 1.0F / h;

    // extract ix, iy, iz
    unsigned ix = p->x[0] / h;
    unsigned iy = p->x[1] / h;
    unsigned iz = p->x[2] / h;
    
    unsigned num_bins = 0;

    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                if (dx == 0 && dy == 0 && dz == 0) 
                    continue;

                unsigned jx = ix + dx;
                unsigned jy = iy + dy;
                unsigned jz = iz + dz;

                // out of boundary
                if (jx < 0 || jx >= XMAX || jy < 0 || jy >= YMAX || jz < 0 || jz >= ZMAX) 
                    continue;
                
                unsigned zm_index = zm_encode(jx & HASH_MASK, jy & HASH_MASK, jz & HASH_MASK);
                buckets[num_bins++] = zm_index;
            }
        }
    }

    return num_bins;
    /* END TASK */
}

void hash_particles(sim_state_t* s, float h)
{
    /* BEGIN TASK */
    memset(s->hash, 0, HASH_SIZE * sizeof(particle_t*));
    for (int i = 0; i < s->n; i++) { // iterate through each particle
        particle_t* pi = &s->part[i];
        unsigned zm_index = particle_bucket(pi, h); // calculate zm index (the hash)
        pi->next = s->hash[zm_index]; // prepend current particle to hash bucket
        s->hash[zm_index] = pi; // update head
    }
    /* END TASK */
}
