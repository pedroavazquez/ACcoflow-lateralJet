/*
 * =====================================================================================
	Immersed coflow jet with AC electric field (axisymmetric, multigrid).

	What this file does:
	- Builds a channel with dimensions (nx=NX, ny=1, nz=1) using multigrid.
	- Runs a two-phase, axisymmetric Navier–Stokes simulation.
	- Optionally solves AC electric field and applies Maxwell stresses.
	- Writes periodic dumps (restart) and snapshots (visualization).

	How to build:
		CC='mpicc -D_MPI=_ -DNX=_ -DLEVEL=_ -DELECTRIC=_ -DDI=_' make lateralJet_params.tst

	Compile-time parameters (fixed at build time):
		MPI       = number of MPI processes
		NX        = number of boxes in x (must equal MPI)
		LEVEL     = initial refinement level
		ELECTRIC  = 1 enable electric field; 0 disable
		DI        = 1 use Robin/impedance BC at top; 0 use Dirichlet
		RESTORE   = 1 restart from dump (reads params.txt); 0 fresh run

	Runtime parameters:
	- Stored in params.txt and read on restore runs.
	- Can be overridden at build time with -D_XXX=value (see apply_compile_time_overrides).
	- You can pass a parameter file at runtime:
	    ./lateralJet_params [params.txt]

	Typical workflow:
	1) Fresh run (RESTORE=0, default):
	   - Parameters initialized from defaults and -D_ overrides.
	   - params.txt is written for record/restart.
	2) Restart run (RESTORE=1 at compile time):
	   - params.txt is read, and DUMP_FILE is used to restore the latest dump.
	   - params.txt is updated with the newest dump name.

	Related tools:
	- readLateralJet.c reads dump files produced by this simulation.
 * =====================================================================================
 */

#include "grid/multigrid.h"
#include "axi.h"
#include "navier-stokes/centered.h"
#include "navier-stokes/double-projection.h"
#include "two-phase.h"
#include "tension.h"
#include "curvature.h"
#include "view.h"
#include "tag.h"
#include "drop_stat.h"
#include "poisson_complex.h"

#define PI (3.14159265359)
#define EPS0        (8.854e-12)   // vacuum permittivity

/*
=================================================
Compile-time parameters (must be set at compile time)
=================================================
*/
#ifndef NX
#define NX 1
#endif
#ifndef LEVEL
#define LEVEL 10
#endif
#ifndef ELECTRIC
#define ELECTRIC 1
#endif
#ifndef DI
#define DI 1
#endif
#ifndef RESTORE
#define RESTORE 0
#endif

/*
=================================================
Runtime parameters structure
=================================================
*/
typedef struct {
	// Geometry
	double R1;          // radius of inlet for inner fluid
	double R2;          // radius of channel and size of box
	double LIN;         // size of radial inlet for outer liquid
	double LBOX;        // size of one box

	char DUMP_FILE[256]; // name of dump file for restore

	// Electrical properties
	double EPSR1;       // dielectric constant for inner fluid
	double EPSR2;       // dielectric constant for outer fluid
	double SIGMA1;      // electrical conductivity for inner fluid
	double SIGMA2;      // electrical conductivity for outer fluid
	double CL;          // external capacity per unit length

	// Viscosity
	double MU1;         // dynamic viscosity of inner fluid
	double MU2;         // dynamic viscosity of outer fluid
	double OH1;         // Ohnesorge number inner
	double OH2;         // Ohnesorge number outer

	// Capillary and velocity
	double CA1;         // capillary number inner fluid
	double CA2;         // capillary number outer fluid

	// Electric field
	double FREQ;        // AC frequency
	double BOE;         // Electric Bond number
	double TELEC;       // time when AC force starts
	double V0;          // non-dimensional applied voltage

	// Surface tension
	double GAMMA;       // surface tension coefficient

	// Time control
	double TEND;        // final simulation time
	double DTOUT;       // snapshots interval
	double DTDUMP;      // dump interval
	double DTMAXMINE;   // maximum time step
} SimParams;

// Global parameters instance
SimParams params;

// Derived quantities (computed from params)
double EPSA1, EPSA2;    // absolute permittivities
double FREQOHM;         // ohmic frequency
double FORCE;           // parameter for AC force
double LX;              // channel length
double AR, BR, CR;      // Robin BC coefficients (real)
double AI, BI, CI;      // Robin BC coefficients (imaginary)

/*
=================================================
Parameter file I/O functions
=================================================
*/
void init_default_params(SimParams *p) {
	// Geometry
	p->R1 = 1.0;
	p->R2 = 1.0;
	p->LIN = 1.0;       // will be set to R2 if not specified
	p->LBOX = 1.0;      // will be set to R2 if not specified

	strcpy(p->DUMP_FILE, "dump");  // default dump file name

	// Electrical
	p->EPSR1 = 80.0;
	p->EPSR2 = 2.1;
	p->SIGMA1 = 3.e-3;
	p->SIGMA2 = 1.e-10;
	p->CL = 0.1;

	// Viscosity
	p->MU1 = 0.001;
	p->MU2 = 0.1;
	p->OH1 = 0.062;
	p->OH2 = 6.2;

	// Capillary and velocity
	p->CA1 = 1.e-4;
	p->CA2 = 0.05;

	// Electric
	p->FREQ = 100.e3;
	p->BOE = 0.0;
	p->TELEC = 0.0;
	p->V0 = 1.0;

	// Surface tension
	p->GAMMA = 1.0;

	// Time control
	p->TEND = 140.0;
	p->DTOUT = 10.0;
	p->DTDUMP = 5.0;
	p->DTMAXMINE = 0.01;
}

void apply_compile_time_overrides(SimParams *p) {
	// Apply compile-time -D overrides for initial runs
	#ifdef _R1
	p->R1 = _R1;
	#endif
	#ifdef _R2
	p->R2 = _R2;
	#endif
	#ifdef _EPSR1
	p->EPSR1 = _EPSR1;
	#endif
	#ifdef _EPSR2
	p->EPSR2 = _EPSR2;
	#endif
	#ifdef _SIGMA1
	p->SIGMA1 = _SIGMA1;
	#endif
	#ifdef _SIGMA2
	p->SIGMA2 = _SIGMA2;
	#endif
	#ifdef _CL
	p->CL = _CL;
	#endif
	#ifdef _MU1
	p->MU1 = _MU1;
	#endif
	#ifdef _MU2
	p->MU2 = _MU2;
	#endif
	#ifdef _OH1
	p->OH1 = _OH1;
	#endif
	#ifdef _OH2
	p->OH2 = _OH2;
	#endif
	#ifdef _CA1
	p->CA1 = _CA1;
	#endif
	#ifdef _CA2
	p->CA2 = _CA2;
	#endif
	#ifdef _FREQ
	p->FREQ = _FREQ;
	#endif
	#ifdef _BOE
	p->BOE = _BOE;
	#endif
	#ifdef _TELEC
	p->TELEC = _TELEC;
	#endif
	#ifdef _V0
	p->V0 = _V0;
	#endif
	#ifdef _GAMMA
	p->GAMMA = _GAMMA;
	#endif
	#ifdef _TEND
	p->TEND = _TEND;
	#endif
	#ifdef _DTOUT
	p->DTOUT = _DTOUT;
	#endif
	#ifdef _DTDUMP
	p->DTDUMP = _DTDUMP;
	#endif
	#ifdef _DTMAXMINE
	p->DTMAXMINE = _DTMAXMINE;
	#endif
}

void compute_derived_params(SimParams *p) {
	// Set dependent geometry defaults
	if (p->LIN <= 0) p->LIN = p->R2;
	if (p->LBOX <= 0) p->LBOX = p->R2;

	// Compute derived quantities
	EPSA1 = p->EPSR1 * EPS0;
	EPSA2 = p->EPSR2 * EPS0;
	FREQOHM = p->SIGMA1 / EPSA1;
	FORCE = 2.0 * p->BOE/p->CA2;
	LX = p->LBOX * NX;

	// Robin BC coefficients
	AR = 1.0;
	BR = p->EPSR2 / p->CL;
	CR = p->V0;
	AI = 1.0;
	BI = p->EPSR2 / p->CL;
	CI = 0.0;
}

int save_params(const char *filename, const SimParams *p) {
	FILE *fp = fopen(filename, "w");
	if (!fp) {
		fprintf(stderr, "Error: Cannot open %s for writing\n", filename);
		return -1;
	}

	fprintf(fp, "# Simulation parameters for lateralJet\n");
	fprintf(fp, "# Compile-time parameters: NX=%d, LEVEL=%d, ELECTRIC=%d, DI=%d\n",
	        NX, LEVEL, ELECTRIC, DI);
	fprintf(fp, "\n# Geometry\n");
	fprintf(fp, "R1 = %.15e\n", p->R1);
	fprintf(fp, "R2 = %.15e\n", p->R2);
	fprintf(fp, "LIN = %.15e\n", p->LIN);
	fprintf(fp, "LBOX = %.15e\n", p->LBOX);

	fprintf(fp, "\n# Electrical properties\n");
	fprintf(fp, "EPSR1 = %.15e\n", p->EPSR1);
	fprintf(fp, "EPSR2 = %.15e\n", p->EPSR2);
	fprintf(fp, "SIGMA1 = %.15e\n", p->SIGMA1);
	fprintf(fp, "SIGMA2 = %.15e\n", p->SIGMA2);
	fprintf(fp, "CL = %.15e\n", p->CL);

	fprintf(fp, "\n# Viscosity\n");
	fprintf(fp, "MU1 = %.15e\n", p->MU1);
	fprintf(fp, "MU2 = %.15e\n", p->MU2);
	fprintf(fp, "OH1 = %.15e\n", p->OH1);
	fprintf(fp, "OH2 = %.15e\n", p->OH2);

	fprintf(fp, "\n# Capillary and velocity\n");
	fprintf(fp, "CA1 = %.15e\n", p->CA1);
	fprintf(fp, "CA2 = %.15e\n", p->CA2);

	fprintf(fp, "\n# Electric field\n");
	fprintf(fp, "FREQ = %.15e\n", p->FREQ);
	fprintf(fp, "BOE = %.15e\n", p->BOE);
	fprintf(fp, "TELEC = %.15e\n", p->TELEC);
	fprintf(fp, "V0 = %.15e\n", p->V0);

	fprintf(fp, "\n# Surface tension\n");
	fprintf(fp, "GAMMA = %.15e\n", p->GAMMA);

	fprintf(fp, "\n# Time control\n");
	fprintf(fp, "TEND = %.15e\n", p->TEND);
	fprintf(fp, "DTOUT = %.15e\n", p->DTOUT);
	fprintf(fp, "DTDUMP = %.15e\n", p->DTDUMP);
	fprintf(fp, "DTMAXMINE = %.15e\n", p->DTMAXMINE);

	fprintf(fp, "\n# Restore\n");
	fprintf(fp, "DUMP_FILE = %s\n", p->DUMP_FILE);

	fclose(fp);
	fprintf(stdout, "# Parameters saved to %s\n", filename);
	return 0;
}

int load_params(const char *filename, SimParams *p) {
	FILE *fp = fopen(filename, "r");
	if (!fp) {
		fprintf(stderr, "Error: Cannot open %s for reading\n", filename);
		return -1;
	}

	char line[256];
	char name[64];
	char str_value[256];
	double value;

	while (fgets(line, sizeof(line), fp)) {
		// Skip comments and empty lines
		if (line[0] == '#' || line[0] == '\n' || line[0] == '\r')
			continue;

		// Try to parse as numeric parameter
		if (sscanf(line, "%63s = %lf", name, &value) == 2) {
			// Match parameter names
			if (strcmp(name, "R1") == 0) p->R1 = value;
			else if (strcmp(name, "R2") == 0) p->R2 = value;
			else if (strcmp(name, "LIN") == 0) p->LIN = value;
			else if (strcmp(name, "LBOX") == 0) p->LBOX = value;
			else if (strcmp(name, "EPSR1") == 0) p->EPSR1 = value;
			else if (strcmp(name, "EPSR2") == 0) p->EPSR2 = value;
			else if (strcmp(name, "SIGMA1") == 0) p->SIGMA1 = value;
			else if (strcmp(name, "SIGMA2") == 0) p->SIGMA2 = value;
			else if (strcmp(name, "CL") == 0) p->CL = value;
			else if (strcmp(name, "MU1") == 0) p->MU1 = value;
			else if (strcmp(name, "MU2") == 0) p->MU2 = value;
			else if (strcmp(name, "OH1") == 0) p->OH1 = value;
			else if (strcmp(name, "OH2") == 0) p->OH2 = value;
			else if (strcmp(name, "CA1") == 0) p->CA1 = value;
			else if (strcmp(name, "CA2") == 0) p->CA2 = value;
			else if (strcmp(name, "FREQ") == 0) p->FREQ = value;
			else if (strcmp(name, "BOE") == 0) p->BOE = value;
			else if (strcmp(name, "TELEC") == 0) p->TELEC = value;
			else if (strcmp(name, "V0") == 0) p->V0 = value;
			else if (strcmp(name, "GAMMA") == 0) p->GAMMA = value;
			else if (strcmp(name, "TEND") == 0) p->TEND = value;
			else if (strcmp(name, "DTOUT") == 0) p->DTOUT = value;
			else if (strcmp(name, "DTDUMP") == 0) p->DTDUMP = value;
			else if (strcmp(name, "DTMAXMINE") == 0) p->DTMAXMINE = value;
		}
		// Try to parse as string parameter
		else if (sscanf(line, "%63s = %255s", name, str_value) == 2) {
			if (strcmp(name, "DUMP_FILE") == 0) {
				strncpy(p->DUMP_FILE, str_value, sizeof(p->DUMP_FILE) - 1);
				p->DUMP_FILE[sizeof(p->DUMP_FILE) - 1] = '\0';
			}
		}
	}

	fclose(fp);
	fprintf(stdout, "# Parameters loaded from %s\n", filename);
	return 0;
}

void print_params(const SimParams *p) {
	fprintf(stdout, "# ===== Simulation Parameters =====\n");
	fprintf(stdout, "# Compile-time: NX=%d, LEVEL=%d, ELECTRIC=%d, DI=%d\n",
	        NX, LEVEL, ELECTRIC, DI);
	fprintf(stdout, "# R1      = %g\n", p->R1);
	fprintf(stdout, "# R2      = %g\n", p->R2);
	fprintf(stdout, "# LIN     = %g\n", p->LIN);
	fprintf(stdout, "# LBOX    = %g\n", p->LBOX);
	fprintf(stdout, "# LX      = %g\n", LX);
	fprintf(stdout, "# EPSR1   = %g\n", p->EPSR1);
	fprintf(stdout, "# EPSR2   = %g\n", p->EPSR2);
	fprintf(stdout, "# SIGMA1  = %g\n", p->SIGMA1);
	fprintf(stdout, "# SIGMA2  = %g\n", p->SIGMA2);
	fprintf(stdout, "# CL      = %g\n", p->CL);
	fprintf(stdout, "# MU1     = %g\n", p->MU1);
	fprintf(stdout, "# MU2     = %g\n", p->MU2);
	fprintf(stdout, "# OH1     = %g\n", p->OH1);
	fprintf(stdout, "# OH2     = %g\n", p->OH2);
	fprintf(stdout, "# CA1     = %g\n", p->CA1);
	fprintf(stdout, "# CA2     = %g\n", p->CA2);
	fprintf(stdout, "# FREQ    = %e\n", p->FREQ);
	fprintf(stdout, "# FREQOHM = %g\n", FREQOHM);
	fprintf(stdout, "# BOE     = %e\n", p->BOE);
	fprintf(stdout, "# FORCE   = %e\n", FORCE);
	fprintf(stdout, "# TELEC   = %g\n", p->TELEC);
	fprintf(stdout, "# V0      = %g\n", p->V0);
	fprintf(stdout, "# GAMMA   = %g\n", p->GAMMA);
	fprintf(stdout, "# TEND    = %g\n", p->TEND);
	fprintf(stdout, "# DTOUT   = %g\n", p->DTOUT);
	fprintf(stdout, "# DTDUMP  = %g\n", p->DTDUMP);
	fprintf(stdout, "# DTMAX   = %g\n", p->DTMAXMINE);
	fprintf(stdout, "# DUMP_FILE = %s\n", p->DUMP_FILE);
	fprintf(stdout, "# ================================\n\n");
}

/*
=================================================
Macros that use runtime parameters
=================================================
*/
// Macro for mixed boundary condition
#define robin(a,b,c) ((dirichlet ((c)*Delta/(2*(b) + (a)*Delta))) + ((neumann (0))*((2*(b) - (a)*Delta)/(2*(b) + (a)*Delta) + 1.)))

// Macros for computing conductivity and dielectric constant values
#define sig(f) ((clamp(f,0,1)*params.SIGMA1 + (1.-clamp(f,0,1))*params.SIGMA2))
#define epsA(f) (clamp(f,0,1)*EPSA1 + (1.-clamp(f,0,1))*EPSA2)

/*
=================================================
Velocity profiles
=================================================
*/
double Uin1 (double y) {
	return params.CA1 * params.MU2 / (params.CA2 * params.MU1);
}

double Uin2 (double x) {
	return -1.0;
}

double computeFrequency ( double t ) {
	if (t< 20 )
		return 1000.E3;
	else if ( t<35)
		return 800.E3;
	else if (t<50 )
		return 600.E3;
	else if (t<65 )
		return 400.E3;
	else if (t<80 )
		return 200.E3;
	else if (t< 95)
		return 100.E3;
	else if (t<110)
		return 80.E3;
	else if (t<135)
		return 60.E3;
	else if (t<150)
		return 40.E3;
	else if (t<165)
		return 20.E3;
	else
		return 10.E3;
}

double computeBOE ( double t ) {
	if (t < 10)
		return 0.0;
	else if (t < 30)
		return 5.0;
	else if (t < 40)
		return 10.0;
	else if (t < 50)
		return 20.0;
	else if (t < 60)
		return 50.0;
	else if (t < 70)
		return 100.0;
	else if (t < 80)
		return 200.0;
	else if (t < 90)
		return 300.0;
	else if (t < 100)
		return 400.0;
	else
		return 500.0;
}

/*
=================================================
Fields
=================================================
*/
mgstats mgPhi;
scalar phiR[], phiI[];
face vector sigma[], epsilonA[];
face vector epsilonRWGF[];
face vector alphaR[], alphaI[];
face vector EfR[];
face vector EfI[];
vector ER[];
vector EI[];
scalar qR[];
scalar qI[];
vector FAC[];
vector FACFORCE[];
scalar kappa[];
vector FC[];

/*
=================================================
Boundary conditions
=================================================
*/
// Note: Boundary conditions use global params variable

u.n[left] = dirichlet(Uin1(y));
u.t[left] = dirichlet(0.0);
u.n[top] = ( x <= params.LIN ? dirichlet(Uin2(x)) : dirichlet(0.));
u.t[top] = dirichlet(0.0);
u.n[right] = neumann(0.0);
u.t[right] = neumann(0.0);
u.n[bottom] = neumann(0.0);
u.t[bottom] = neumann(0.0);

p[right] = dirichlet(0.);
pf[right] = dirichlet(0.);

phiR[left] = dirichlet(0.);
phiI[left] = dirichlet(0.);
phiR[right] = neumann(0);
phiI[right] = neumann(0);
phiR[bottom] = neumann(0);
phiI[bottom] = neumann(0);
#if DI
phiR[top] = (x <= params.LIN ? neumann(0) : robin(AR, BR, CR));
phiI[top] = (x <= params.LIN ? neumann(0.) : robin(AI, BI, CI));
#else
phiR[top] = (x <= params.LIN ? neumann(0) : dirichlet(params.V0));
phiI[top] = (x <= params.R2 ? neumann(0) : dirichlet(0));
#endif

f[left] = (y < params.R1 ? 1.0 : 0.);
f[top] = 0.;

double xFront = 0.;
double delMax = 0.;
double fc_min = 0.;
double fc_max = 0.;
int fc_scale_set = 0;

// Timing variables for dumps and snapshots (initialized in event init)
double t_next_dump = 0.;
double t_next_snapshot = 0.;

void save_radial_profiles_per_pid (double t, double dx_ref, double dy_ref, scalar f)
{
	int rank = pid();
	double x_min = rank * params.LBOX;
	double x_max = (rank + 1.) * params.LBOX;
	double y_min = 0.;
	double y_max = params.R2 - delMax;

	if (dx_ref <= 0.)
		dx_ref = params.LBOX;

	int nlines_x = max(1, (int) (params.LBOX/dx_ref + 0.5));
	int nsamples_y = max(2, (int) ((y_max - y_min)/dy_ref + 0.5) + 1);

	char filename[256];
	sprintf(filename, "radial_profiles_pid%04d_t%08.4f.dat", rank, t);

	FILE *fp = fopen(filename, "w");
	if (!fp) {
		fprintf(stderr, "Error: Cannot open %s for writing (pid=%d)\n", filename, rank);
		return;
	}

	fprintf(fp, "# Radial profiles sampled from restored dump\n");
	fprintf(fp, "# pid = %d\n", rank);
	fprintf(fp, "# x_min = %.15e\n", x_min);
	fprintf(fp, "# x_max = %.15e\n", x_max);
	fprintf(fp, "# y_min = %.15e\n", y_min);
	fprintf(fp, "# y_max = %.15e\n", y_max);
	fprintf(fp, "# nlines_x = %d\n", nlines_x);
	fprintf(fp, "# nsamples_y = %d\n", nsamples_y);
	fprintf(fp, "# Columns:\n");
	fprintf(fp, "# 1:line_id 2:x 3:y 4:f 5:ux 6:uy 7:phiR 8:phiI 9:ERx 10:ERy 11:EIx 12:EIy 13:qR 14:qI 15:FCx 16:FCy 17:FACx 18:FACy 19:FACFORCEx 20:FACFORCEy\n");

	for (int i = 0; i < nlines_x; i++) {
		double x_line = x_min + (i + 0.5) * (x_max - x_min)/nlines_x;
		fprintf(fp, "# line_id %d x = %.15e\n", i, x_line);

		for (int j = 0; j < nsamples_y; j++) {
			double y_line = y_min + (double) j * (y_max - y_min)/(nsamples_y - 1);

			double f_v = interpolate(f, x_line, y_line);
			double ux_v = interpolate(u.x, x_line, y_line);
			double uy_v = interpolate(u.y, x_line, y_line);
			double phiR_v = interpolate(phiR, x_line, y_line);
			double phiI_v = interpolate(phiI, x_line, y_line);
			double ERx_v = interpolate(ER.x, x_line, y_line);
			double ERy_v = interpolate(ER.y, x_line, y_line);
			double EIx_v = interpolate(EI.x, x_line, y_line);
			double EIy_v = interpolate(EI.y, x_line, y_line);
			double qR_v = interpolate(qR, x_line, y_line);
			double qI_v = interpolate(qI, x_line, y_line);
			double FCx_v = interpolate(FC.x, x_line, y_line);
			double FCy_v = interpolate(FC.y, x_line, y_line);
			double FACx_v = interpolate(FAC.x, x_line, y_line);
			double FACy_v = interpolate(FAC.y, x_line, y_line);
			double FACFORCEx_v = interpolate(FACFORCE.x, x_line, y_line);
			double FACFORCEy_v = interpolate(FACFORCE.y, x_line, y_line);

			fprintf(fp,
			        "%d %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
			        i, x_line, y_line, f_v, ux_v, uy_v, phiR_v, phiI_v,
			        ERx_v, ERy_v, EIx_v, EIy_v, qR_v, qI_v, FCx_v, FCy_v,
			        FACx_v, FACy_v, FACFORCEx_v, FACFORCEy_v);
		}
		fprintf(fp, "\n\n");
	}

	fclose(fp);
	fprintf(stdout, "# Saved radial profiles: %s (pid=%d, x:[%g, %g])\n",
	        filename, rank, x_min, x_max);
}

void save_axis_profiles ()
{

	FILE *fp = fopen("axis_profile.dat", "w");
	if (fp) {
		fprintf(fp, "# Axis profile sampled from restored dump\n");
		fprintf(fp, "# 1:x 2:phiR 3:ux 4:ERx 5:f\n");
		for (double x = 0; x <= LX-delMax; x += 0.2*delMax	) {
			double phiR_v = interpolate(phiR, x, 0.);
			double ux_v = interpolate(u.x, x, 0.);
			double ERx_v = interpolate(ER.x, x, 0.);
			double f_v = interpolate(f, x, 0.);
			fprintf(fp, "%.15e %.15e %.15e %.15e %.15e\n", x, phiR_v, ux_v, ERx_v, f_v);
		}
		fclose(fp);
		fprintf(stdout, "# Saved axis profile: axis_profile.dat\n");
	} else {
		fprintf(stderr, "Error: Cannot open axis_profile.dat for writing\n");
	}

}

/*
=================================================
Main
=================================================
*/
int main (int argc, char * argv[])
{
	const char *params_file = NULL;
	const char *dump_file = NULL;

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
			params_file = argv[++i];
		} else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
			dump_file = argv[++i];
		} else {
			fprintf(stderr, "Usage: %s -p <params.txt> -d <dump-file>\n", argv[0]);
			return 1;
		}
	}

	if (!params_file || !dump_file) {
		fprintf(stderr, "Usage: %s -p <params.txt> -d <dump-file>\n", argv[0]);
		return 1;
	}

	// Initialize parameters with defaults
	init_default_params(&params);

	// Apply compile-time overrides (if any -D_PARAM=value flags were used)
	apply_compile_time_overrides(&params);

	// Load parameters from file.
	if (load_params(params_file, &params) != 0) {
		fprintf(stderr, "Error: Could not load %s\n", params_file);
		return 1;
	}

	strncpy(params.DUMP_FILE, dump_file, sizeof(params.DUMP_FILE) - 1);
	params.DUMP_FILE[sizeof(params.DUMP_FILE) - 1] = '\0';

	// Compute derived quantities
	compute_derived_params(&params);

	// Print all parameters
	print_params(&params);

	dimensions ( nx = NX, ny = 1, nz = 1);
	size ( LX );
	init_grid ( 1 << LEVEL );
	origin ( 0., 0.);

	rho1 = 1., rho2 = 1.;
	mu1 = params.MU1/params.MU2, mu2 = 1.;
	f.sigma = params.GAMMA/params.CA2;
	stokes = true;

	TOLERANCE = 1.e-6 [*];
	DT = params.DTMAXMINE;

	params.TEND=0.;
	run();

	return 0;
}

event init ( t = 0 )
{

	fprintf(stdout, "# Restoring from: %s\n", params.DUMP_FILE);
	restore(file = params.DUMP_FILE);

	// Restore values of metric factors (essential in axi)
	foreach()
		cm[] = y;
	cm[top] = dirichlet(y);
	cm[bottom] = dirichlet(y);
	foreach_face()
		fm.x[] = max(y, 1./HUGE);
	fm.t[top] = dirichlet(y);
	fm.t[bottom] = dirichlet(y);

	f.prolongation = refine_bilinear;
	boundary({f});


	fprintf(stdout, "# Restore complete. No simulation steps will be run.\n");

	// Find Delta
	scalar del[];
	foreach()
		del[] = Delta;
	delMax = statsf(del).max;

	fprintf(stdout, "# delMax  = %g\n\n", delMax);

#if 0
	// Compute electric field and related quantities from restored phiR and phiI
	foreach_face() {
		double f0 = 0.5*(f[] + f[-1]);
		sigma.x[] = sig(f0);
		epsilonA.x[] = epsA(f0);
		epsilonRWGF.x[] = epsilonA.x[]*fm.x[]/EPS0;
		alphaR.x[] = sigma.x[]*fm.x[];
		alphaI.x[] = 2.*PI*params.FREQ*epsilonA.x[]*fm.x[];
	}

	mgPhi = poisson_complex(phiR, phiI, alphaR, alphaI, tolerance = 1.e-6 );

	foreach_face() {
		EfR.x[] = -(phiR[] - phiR[-1])/Delta;
		EfI.x[] = -(phiI[] - phiI[-1])/Delta;
	}

	foreach() {
		foreach_dimension() {
			ER.x[] = (EfR.x[] + EfR.x[1]) / 2.0;
			EI.x[] = (EfI.x[] + EfI.x[1]) / 2.0;
		}
	}

	foreach() {
		double div_Dr = 0.;
		double div_Di = 0.;

		foreach_dimension() {
			double Dr_right = epsilonRWGF.x[1] * EfR.x[1];
			double Di_right = epsilonRWGF.x[1] * EfI.x[1];

			double Dr_left  = epsilonRWGF.x[] * EfR.x[];
			double Di_left  = epsilonRWGF.x[] * EfI.x[];

			div_Dr += (Dr_right - Dr_left);
			div_Di += (Di_right - Di_left);
		}

		qR[] = div_Dr / (Delta * cm[]);
		qI[] = div_Di / (Delta * cm[]);
	}

	assert (dimension <= 2);
	vector f_vec[];

	foreach_dimension() {
		face vector Mx[];

		foreach_face(x) {
			double sq_E_norm_r = sq(phiR[] - phiR[-1,0]);
			double sq_E_tang_r = sq(phiR[0,1] - phiR[0,-1] + phiR[-1,1] - phiR[-1,-1])/16.;
			double sq_E_norm_i = sq(phiI[] - phiI[-1,0]);
			double sq_E_tang_i = sq(phiI[0,1] - phiI[0,-1] + phiI[-1,1] - phiI[-1,-1])/16.;

			Mx.x[] = 0.5 * epsilonRWGF.x[]/2. *
			         ( (sq_E_norm_r + sq_E_norm_i) - (sq_E_tang_r + sq_E_tang_i) ) / sq(Delta);
		}

		foreach_face(y) {
			double cross_r = (phiR[] - phiR[0,-1]) *
			                 (phiR[1,0] - phiR[-1,0] + phiR[1,-1] - phiR[-1,-1]);
			double cross_i = (phiI[] - phiI[0,-1]) *
			                 (phiI[1,0] - phiI[-1,0] + phiI[1,-1] - phiI[-1,-1]);

			Mx.y[] = 0.5 * epsilonRWGF.y[] * (cross_r + cross_i) / sq(2.*Delta);
		}

		foreach()
			f_vec.x[] = (Mx.x[1,0] - Mx.x[] + Mx.y[0,1] - Mx.y[])/(Delta*cm[]);
	}

#if AXI
	foreach() {
		double mod_sq_r = sq(phiR[1,0] - phiR[-1,0]) + sq(phiR[0,1] - phiR[0,-1]);
		double mod_sq_i = sq(phiI[1,0] - phiI[-1,0]) + sq(phiI[0,1] - phiI[0,-1]);

		f_vec.y[] += (mod_sq_r + mod_sq_i)/(16.*cm[]*sq(Delta))
		  *(epsilonRWGF.x[]/fm.x[] + epsilonRWGF.y[]/fm.y[] +
		    epsilonRWGF.x[1,0]/fm.x[1,0] + epsilonRWGF.y[0,1]/fm.y[0,1])/4.;
	}
#endif

	foreach()
		foreach_dimension() {
			FAC.x[] = f_vec.x[];
			FACFORCE.x[] = FORCE*(f_vec.x[]);
		}

	curvature (f, kappa);

	double sigma_val = f.sigma;
	foreach() {
		foreach_dimension() {
			double gradf = (f[1] - f[-1])/(2.*Delta);
			FC.x[] = sigma_val*kappa[]*gradf;
		}
	}
#endif

	//save_radial_profiles_per_pid(t, delMax);
	save_radial_profiles_per_pid(t, 0.2*params.LBOX, 0.2*delMax, 	f);

	// save axis profiles
	save_axis_profiles();

	// save facets
	{
		char filename[256];
		sprintf(filename, "facets-%04d.dat", pid());

		FILE *ff = fopen(filename, "w");
		if (ff) {
			output_facets(f, ff);
		} else {
			fprintf(stderr, "Error: Cannot open %s for writing\n", filename);
		}
		fclose(ff);
		fprintf(stdout, "# Saved facets for visualization: %s\n", filename);
	}

	// Save fields for visualization
	{
		FILE *fp = fopen("fields.dat", "w");
		if (fp) {
			output_field({f, u, phiR, phiI, ER, EI, qR, qI, FC, FAC, FACFORCE},fp, 5*N,bilinear);}
		else {
			fprintf(stderr, "Error: Cannot open fieldsdat for writing\n");
		}
		fclose(fp);
		fprintf(stdout, "# Saved fields for visualization: fields.dat\n");
	}

	// Jet radius along axis (equivalent-volume radius, robust to interface noise)
	{
		double x_min = 0.;
		double x_max = NX * params.LBOX;
		double y_lo = 0.5*delMax;
		double y_hi = params.R2 - 0.5*delMax;
		double dx_radius = delMax;
		double dy_radius = delMax;
		int nx_lines = max(1, (int) ((x_max - x_min)/dx_radius + 0.5));
		int ny_samples = max(2, (int) ((y_hi - y_lo)/dy_radius + 0.5) + 1);
		double dy_line = (y_hi - y_lo)/(ny_samples - 1);

		char radius_name[256];
		sprintf(radius_name, "jet_radius_eq.dat");
		FILE *fr = fopen(radius_name, "w");
		if (!fr) {
			fprintf(stderr, "Error: Cannot open %s for writing\n", radius_name);
		}
		else {
			fprintf(fr, "# Jet radius profile from restored dump\n");
			fprintf(fr, "# Definition: equivalent-volume radius from f(y)\n");
			fprintf(fr, "# Planar: radius = integral_0^R2 f dy\n");
			fprintf(fr, "# Axisymmetric: radius = sqrt(2*integral_0^R2 y*f dy)\n");
			fprintf(fr, "# x-range: [0, LX]\n");
			fprintf(fr, "# 1:x 2:radius\n");

			for (int i = 0; i < nx_lines; i++) {
				double x_line = x_min + (i + 0.5)*(x_max - x_min)/nx_lines;
				double int_f = 0.;
				double int_yf = 0.;

				for (int j = 0; j < ny_samples; j++) {
					double y_line = y_lo + (double) j * (y_hi - y_lo)/(ny_samples - 1);
					double f_v = clamp(interpolate(f, x_line, y_line), 0., 1.);
					int_f += f_v * dy_line;
					int_yf += y_line * f_v * dy_line;
				}

				double radius = int_f; // planar default
#if AXI
				radius = sqrt(max(0., 2.*int_yf));
#endif

				fprintf(fr, "%.15e %.15e\n", x_line, radius);
			}
			fclose(fr);
			fprintf(stdout, "# Saved jet radius profile: %s\n", radius_name);
		}
	}
	fprintf(stdout, "f[20] at axis = %g\n", interpolate(f, 20., 0.));
	exit(0);



	char label[200];
	view( tx = -1.5, ty = -0.3, sx = 4.0, sy = 8.0, width = 1200, height = 300 );

	// u.x
	box();
	sprintf(label, "Ux t:%2.1f", t);
	squares("u.x", min = -3., max = 10., cbar = true, pos = {0.5, -0.2}, levels = 10, mid = true, format = "%6.4f", label = label);
	draw_vof("f", lw = 2 );
	save("ux.png");

	// phiR
	clear();
	box();
	sprintf(label, "phiR t:%2.1f", t);
	squares("phiR", min = 0, max = statsf(phiR).max, cbar = true, pos = {0.5, -0.2}, levels = 10, mid = true, format = "%6.4f", label = label);
	draw_vof("f", lw = 2 , lc={1,1,1});
	save("phiR.png");

	// ERy
	clear();
	box();
	sprintf(label, "ERy t:%2.1f", t);
	squares("ER.y", min = 0, max = statsf(phiR).max, cbar = true, pos = {0.5, -0.2}, levels = 10, mid = true, format = "%6.4f", label = label);
	draw_vof("f", lw = 2 , lc={1,1,1});
	save("ERy.png");

	// FCx and FCy for x > LIN
	scalar FCx_mask[], FCy_mask[];
	scalar FACFORCEx_mask[], FACFORCEy_mask[];
	foreach() {
		if (x > params.LIN) {
			FCx_mask[] = FC.x[];
			FCy_mask[] = FC.y[];
			FACFORCEx_mask[] = FACFORCE.x[];
			FACFORCEy_mask[] = FACFORCE.y[];
		} else {
			FCx_mask[] = nodata;
			FCy_mask[] = nodata;
			FACFORCEx_mask[] = nodata;
			FACFORCEy_mask[] = nodata;
		}
	}

	if (!fc_scale_set) {
		stats s1 = statsf(FCx_mask);
		stats s2 = statsf(FCy_mask);
		fc_min = min(s1.min, s2.min);
		fc_max = max(s1.max, s2.max);
		if (fc_min == fc_max) {
			fc_min = -1e-12;
			fc_max =  1e-12;
		}
		fc_scale_set = 1;
	}

	// FCx
	clear();
	box();
	sprintf(label, "FCx (x>LIN) t:%2.1f", t);
	squares("FCx_mask", min = fc_min, max = fc_max, cbar = true, pos = {0.5, -0.2}, levels = 10, mid = true, format = "%6.4f", label = label);
	draw_vof("f", lw = 2 , lc={0,0,0});
	save("FCx.png");

	// FCy
	clear();
	box();
	sprintf(label, "FCy (x>LIN) t:%2.1f", t);
	squares("FCy_mask", cbar = true, pos = {0.5, -0.2}, levels = 10, mid = true, format = "%6.4f", label = label);
	draw_vof("f", lw = 2 , lc={0,0,0});
	save("FCy.png");

	// FACFORCEx
	clear();
	box();
	sprintf(label, "FACFORCEx (x>LIN) t:%2.1f", t);
	squares("FACFORCEx_mask", cbar = true, pos = {0.5, -0.2}, levels = 10, mid = true, format = "%6.4f", label = label);
	draw_vof("f", lw = 2 , lc={0,0,0});
	save("FACFORCEx.png");

	// FACFORCEy
	clear();
	box();
	sprintf(label, "FACFORCEy (x>LIN) t:%2.1f", t);
	squares("FACFORCEy_mask", cbar = true, pos = {0.5, -0.2}, levels = 10, mid = true, format = "%6.4f", label = label);
	draw_vof("f", lw = 2 , lc={0,0,0});
	save("FACFORCEy.png");

}
