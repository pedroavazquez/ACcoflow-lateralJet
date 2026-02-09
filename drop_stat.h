/*
 * =====================================================================================
 *
 *       Filename:  drop_stat.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  25/02/19 12:06:10
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *        Company:  
 *
 * =====================================================================================
 */
/**
# Useful functions that uses *tag.h*. 

This code is a copy from [lopez](http://basilisk.fr/sandbox/lopez/droplet_stat.h), all credit to
him */

static double remove_droplets_vol (scalar c, int nr, bool droplet) {

  /**
    We can remove droplets (converting *c* = 1 to *c* = 0) or bubbles. The
    boolean variable below control this issue. If the withdrawal of
    droplets/bubbles is to be restricted to some part of the computational
    domain the definition of the scalar *c[]* below could be altered. */

  scalar m[];
  double vol = 0., THR = 1e-2; //THRESHOLD
  foreach()
    m[] = (droplet ? (c[] > THR) : (c[] < (1.-THR)));
  int n = tag (m);

  /** 
    The event is created to remove tiny bubbles or droplets. Since 
    The tag *m[]* is not ordered by size, we need to calculate the
    size of the droplet/bubbles before proceed with the (ordered)
    removing.  

    We set the number *nr* of droplets/bubbles that shall
    remain after the event. They must be the *nr* largest. Naturally,
    the removing proceeds if the number of droplets/bubbles *n*
    is larger than the number of them we wish to keep, *nr*. */

  if (n > nr) {
    double v[n];
    for (int j = 0; j < n; j++)
      v[j] = 0.;
    foreach_leaf()
      if (m[] > 0) {
        int j = m[] - 1;
        v[j] += dv()*(droplet ? c[] : 1. - c[]);
      }

#if _MPI
    MPI_Allreduce (MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

    /**
      Some sorting procedure has to be done to identify the $nr^{th}$
      largest volume, $vmax$. We will use for its simplicity the <a
      href="http://en.wikipedia.org/wiki/Bubble_sort"> bubble sorting
      scheme</a>. The sorting will be done over the auxiliary array *vs*.*/

    double vs[n];
    for (int j = 0; j < n; j++)
      vs[j] = v[j];

    int nl = n;
    do {
      int newn = 0;
      for (int i = 1; i < nl; i++) 
        if(vs[i-1] > vs[i]) {
          double aux = vs[i];
          vs[i] = vs[i-1];
          vs[i-1] = aux;
          newn = i;
        }
      nl = newn;
    } while (nl > n-nr);

    double vmax = vs[n-nr];

    for(int j = 0; j < nr; j++)
      vol += vs[j];

    /**
      Once the threshold value *vmax* is computed the droplets with
      volume below this threshold should be removed. */

    foreach() 
      if (m[] > 0) {
        int j = m[]-1;
        if(v[j] < vmax)
          c[] = droplet ? 0. : 1.;
      }
    boundary ({c});
  }
  return vol;
}


static double remove_droplets_volmin (scalar c, double volmin) {

  /**
    We remove droplets of volume<volmin. */

  scalar m[];
  double THR = 1e-2; //THRESHOLD
  foreach()
    m[] = (c[] > THR);
  int n = tag (m);

  /** 
    The event is created to remove tiny droplets. Since 
    The tag *m[]* is not ordered by size, we need to calculate the
    size of the droplet/bubbles before proceed with the 
    removing.  
    */

  double v[n]; // volume of each neighborhood
  for (int j = 0; j < n; j++)
    v[j] = 0.;
  foreach_leaf()
    if (m[] > 0) {
      int j = m[] - 1;
      v[j] += dv()*(c[]);
    }

#if _MPI
  MPI_Allreduce (MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  /**
    Remove the droplets with volume smaller than volmin. It also computes the X
    coordinate of the tip of neighborhood with tag 1
    */

  scalar ff[];
  foreach() {
    if (m[] > 0) {
      int j = m[]-1;
      if(v[j] < volmin)
        c[] =  0.;
    }
    ff[] = ( m[]==1 ? c[] : 0.);
  }
  boundary ({c,ff});

  scalar pos[];
  position(ff, pos, {1,0});

  return statsf(pos).max;
#if 0
  /**
   Compute the X coordinate of the tip of the jet
   */
  foreach()
    ff[] = ( m[]==1 ? c[] : 0.);
  boundary({ff});
#endif

}


