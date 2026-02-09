/**
   Complex Poisson solver (lambda = 0, rho = 0)

   Solve for phi = phir + i phii:
     div( (sigma + i*omega*epsilon) grad(phi) ) = 0

   The complex coefficient is provided on faces as two face vectors:
     sigmar = sigma
     sigmai = omega*epsilon   (for exp(+i*omega t))

   If you use exp(-i*omega t), use sigmai = -omega*epsilon instead.
*/

#ifndef POISSON_COMPLEX_H
#define POISSON_COMPLEX_H

#include "poisson.h"

#if EMBED
#warning "poisson_complex: EMBED flux terms are not implemented here (rho=0, lambda=0 only)."
#endif

struct PoissonComplex {
  scalar ar, ai;               // solution (real, imag)
  (const) face vector alphar;  // real part of complex coefficient on faces
  (const) face vector alphai;  // imag part of complex coefficient on faces
  double tolerance;
  int nrelax, minlevel;
  scalar * res;                // optional residual storage (two scalars)
};

static inline void cdiv (double nr, double ni,
                         double dr, double di,
                         double * qr, double * qi)
{
  double den = dr*dr + di*di;
  if (den == 0.) { *qr = 0.; *qi = 0.; return; }
  *qr = (nr*dr + ni*di)/den;
  *qi = (ni*dr - nr*di)/den;
}

/**
Relaxation for the correction equation:
  div( alpha grad(da) ) = res
with alpha complex, res complex.
*/
static void relax_complex (scalar * al, scalar * bl, int l, void * data)
{
  scalar ar = al[0], ai = al[1];   // correction fields (dar, dai)
  scalar rr = bl[0], ri = bl[1];   // residual fields (real, imag)

  struct PoissonComplex * p = (struct PoissonComplex *) data;
  (const) face vector alphar = p->alphar;
  (const) face vector alphai = p->alphai;

#if JACOBI
  scalar cr[], ci[];
#else
  scalar cr = ar, ci = ai;
#endif

#if GAUSS_SEIDEL || _GPU
  for (int parity = 0; parity < 2; parity++)
    foreach_level_or_leaf (l, nowarning)
      if (level == 0 || ((point.i + parity) % 2) != (point.j % 2))
#else
  foreach_level_or_leaf (l, nowarning)
#endif
  {
    // Complex numerator N = -Delta^2 * res + sum_faces alpha_face * a_neighbor
    double Nr = -sq(Delta)*rr[];
    double Ni = -sq(Delta)*ri[];

    // Complex denominator D = sum_faces alpha_face  (lambda = 0)
    double Dr = 0., Di = 0.;

    foreach_dimension() {
      // + face contribution
      double apr = alphar.x[1], api = alphai.x[1];
      double urp = ar[1],      uip = ai[1];
      Nr += apr*urp - api*uip;
      Ni += apr*uip + api*urp;
      Dr += apr;
      Di += api;

      // - face contribution
      double amr = alphar.x[], ami = alphai.x[];
      double urm = ar[-1],     uim = ai[-1];
      Nr += amr*urm - ami*uim;
      Ni += amr*uim + ami*urm;
      Dr += amr;
      Di += ami;
    }

    double qr, qi;
    cdiv (Nr, Ni, Dr, Di, &qr, &qi);
    cr[] = qr;
    ci[] = qi;
  }

#if JACOBI
  foreach_level_or_leaf (l) {
    ar[] = (ar[] + 2.*cr[])/3.;
    ai[] = (ai[] + 2.*ci[])/3.;
  }
#endif
}

/**
Residual: res = b - div(alpha grad(a)), with b=0 here.
We return max |res| over the domain.
*/
static double residual_complex (scalar * al, scalar * bl, scalar * resl, void * data)
{
  (void) bl; // b is identically zero; not used
  scalar ar = al[0], ai = al[1];
  scalar rr = resl[0], ri = resl[1];

  struct PoissonComplex * p = (struct PoissonComplex *) data;
  (const) face vector alphar = p->alphar;
  (const) face vector alphai = p->alphai;

  double maxres = 0.;

#if TREE
  face vector fr[], fi[];
  foreach_face() {
    double dar = face_gradient_x (ar, 0);
    double dai = face_gradient_x (ai, 0);
    double ar_f = alphar.x[], ai_f = alphai.x[];
    fr.x[] = ar_f*dar - ai_f*dai;
    fi.x[] = ar_f*dai + ai_f*dar;
  }

  foreach (reduction(max:maxres), nowarning) {
    double rre = 0., rim = 0.;
    foreach_dimension() {
      rre -= (fr.x[1] - fr.x[])/Delta;
      rim -= (fi.x[1] - fi.x[])/Delta;
    }
    rr[] = rre;
    ri[] = rim;
    double mag = sqrt (rre*rre + rim*rim);
    if (mag > maxres) maxres = mag;
  }

#else // !TREE
  foreach (reduction(max:maxres), nowarning) {
    double rre = 0., rim = 0.;

    foreach_dimension() {
      // face 0
      double dar0 = face_gradient_x (ar, 0);
      double dai0 = face_gradient_x (ai, 0);
      double ar0  = alphar.x[0], ai0 = alphai.x[0];
      double fr0  = ar0*dar0 - ai0*dai0;
      double fi0  = ar0*dai0 + ai0*dar0;

      // face 1
      double dar1 = face_gradient_x (ar, 1);
      double dai1 = face_gradient_x (ai, 1);
      double ar1  = alphar.x[1], ai1 = alphai.x[1];
      double fr1  = ar1*dar1 - ai1*dai1;
      double fi1  = ar1*dai1 + ai1*dar1;

      rre += (fr0 - fr1)/Delta;
      rim += (fi0 - fi1)/Delta;
    }

    rr[] = rre;
    ri[] = rim;
    double mag = sqrt (rre*rre + rim*rim);
    if (mag > maxres) maxres = mag;
  }
#endif

  return maxres;
}

/**
User interface: solves div( (alphar + i alphai) grad(phi) ) = 0
for phi = ar + i ai.
*/
trace
mgstats poisson_complex (scalar ar, scalar ai,
                         (const) face vector alphar = {{-1}},
                         (const) face vector alphai = {{-1}},
                         double tolerance = 0.,
                         int nrelax = 4,
                         int minlevel = 0,
                         scalar * res = NULL)
{
  // defaults: alpha = 1 + i 0
  if (alphar.x.i < 0)
    alphar[] = {1.,1.,1.};
  if (alphai.x.i < 0)
    alphai[] = {0.,0.,0.};

  restriction ((scalar *){alphar, alphai});

  double defaultol = TOLERANCE;
  if (tolerance)
    TOLERANCE = tolerance;

  // b = 0 + i 0
  scalar br[], bi[];
  foreach() br[] = bi[] = 0.;
  boundary ({br, bi});

  struct PoissonComplex p = {ar, ai, alphar, alphai, tolerance, nrelax, minlevel, res};

  mgstats s = mg_solve ({ar, ai}, {br, bi},
                       residual_complex, relax_complex, &p,
                       nrelax, res, max(1, minlevel));

  if (tolerance)
    TOLERANCE = defaultol;

  return s;
}

#endif // POISSON_COMPLEX_H
