/*
 * =====================================================================================
	Reads a simulation of immersed jet with AC field applied

	RESTORE = 1 by default. 

	Build a channel using multigrid with dimensions (nx=NX, ny=1, nz=1)

	Save snapshots in dump files. These file can be read with readLateralJet.c

	This version saves/loads parameters from a text file for restore runs.

	Usage:
		CC='mpicc -D_MPI=_ -DNX=_ -DLEVEL=_ -DELECTRIC=_ -DDI=_' make lateralJet_params.tst

		Compile-time parameters (cannot be changed at runtime):
			MPI 			= number of processes
			NX				= number of boxes (must be = MPI) (1)
			LEVEL			= Refinement level
			ELECTRIC  = 1 -> electric computations; 0 -> no electric comp.
			DI				= Uses distributed impedance boundary condition (0 no DI; 1 DI)

		Runtime parameters (read from params.txt on restore, or set via command line):
			Use command line: ./lateralJet_params [params.txt]
			Or compile with -D flags for initial values

 * =====================================================================================
 */

#include "grid/multigrid.h"
#include "axi.h"
#include "navier-stokes/centered.h"
#include "navier-stokes/double-projection.h"
#include "two-phase.h"
#include "tension.h"
#include "curvature.h"
#include "heights.h"
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

	// Restore flag
	int RESTORE;        // 0 = fresh start, 1 = restore from dump
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
	double U1;          // ratio of average velocity inner/outer
	double U2;          // average velocity outer fluid

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

	// Restore
	p->RESTORE = 1;
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
	p->U1 = 0.0;        // will be computed from CA1/OH1 if 0
	p->U2 = 0.0;        // will be computed from CA2/OH2 if 0

	// Electric
	p->FREQ = 100.e3;
	p->BOE = 0.0;
	p->TELEC = 0.0;
	p->V0 = 1.0;

	// Surface tension
	p->GAMMA = 1.0;

	// Time control
	p->TEND = 165.0;
	p->DTOUT = 2.0;
	p->DTDUMP = 10.0;
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
	#ifdef _U1
	p->U1 = _U1;
	#endif
	#ifdef _U2
	p->U2 = _U2;
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
	#ifdef _RESTORE
	p->RESTORE = _RESTORE;
	#endif
}

void compute_derived_params(SimParams *p) {
	// Set dependent geometry defaults
	if (p->LIN <= 0) p->LIN = p->R2;
	if (p->LBOX <= 0) p->LBOX = p->R2;

	// Compute velocities if not explicitly set
	if (p->U1 <= 0) p->U1 = p->CA1 / p->OH1;
	if (p->U2 <= 0) p->U2 = p->CA2 / p->OH2;

	// Compute derived quantities
	EPSA1 = p->EPSR1 * EPS0;
	EPSA2 = p->EPSR2 * EPS0;
	FREQOHM = p->SIGMA1 / EPSA1;
	FORCE = 2.0 * p->BOE;
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
	fprintf(fp, "U1 = %.15e\n", p->U1);
	fprintf(fp, "U2 = %.15e\n", p->U2);

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
			else if (strcmp(name, "U1") == 0) p->U1 = value;
			else if (strcmp(name, "U2") == 0) p->U2 = value;
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
	fprintf(stdout, "# U1      = %g\n", p->U1);
	fprintf(stdout, "# U2      = %g\n", p->U2);
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
	return params.U1;
}

double Uin2 (double x) {
	return -params.U2;
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

void compute_electric_field() {

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
}

void compute_electric_charge(face vector EfR, face vector EfI, scalar qR, scalar qI) {

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
}

void compute_electric_force() {

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
}

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

/*
=================================================
Main
=================================================
*/
int main (int argc, char * argv[])
{
	// Initialize parameters with defaults
	init_default_params(&params);

	// Apply compile-time overrides (if any -D_PARAM=value flags were used)
	apply_compile_time_overrides(&params);

	// Check for RESTORE flag (can be set via -D_RESTORE=1)
	#ifdef _RESTORE
	params.RESTORE = _RESTORE;
	#endif

	// Handle command line argument for params file
	const char *params_file = "params.txt";
	if (argc > 1) {
		params_file = argv[1];
	}

	if (params.RESTORE) {
		// Load parameters from file for restore run
		if (load_params(params_file, &params) != 0) {
			fprintf(stderr, "Warning: Could not load params, using defaults/compile-time values\n");
		}
	}

	// Compute derived quantities
	compute_derived_params(&params);

	// Print all parameters
	print_params(&params);

	// Save parameters (useful for fresh runs, overwrites for restore)
	if (!params.RESTORE) {
		save_params(params_file, &params);
	}

	dimensions ( nx = NX, ny = 1, nz = 1);
	size ( LX );
	init_grid ( 1 << LEVEL );
	origin ( 0., 0.);

	rho1 = 1., rho2 = 1.;
	mu1 = params.OH1, mu2 = params.OH2;
	f.sigma = params.GAMMA;
	stokes = true;

	TOLERANCE = 1.e-5 [*];
	DT = params.DTMAXMINE;

	run();
}

event init ( t = 0 )
{
	// Find Delta
	scalar del[];
	foreach()
		del[] = Delta;
	delMax = statsf(del).max;

	fprintf(stdout, "# delMax  = %g\n\n", delMax);

	if (!params.RESTORE) {
		// Start from scratch
		foreach() {
			u.x[] = 0.;
			u.y[] = 0.;
			f[] = ( (y < 0.5*params.R1) && (x < params.R1) ? 1.0 : 0);
		}
	} else {
		// Restore from dump file
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
	}

	// Initialize timing for dumps and snapshots
	// For restore runs, calculate next times based on current t
	if (params.RESTORE && t > 0) {
		t_next_dump = (floor(t / params.DTDUMP) + 1) * params.DTDUMP;
		t_next_snapshot = (floor(t / params.DTOUT) + 1) * params.DTOUT;
	} else {
		t_next_dump = params.DTDUMP;
		t_next_snapshot = params.DTOUT;
	}
	fprintf(stdout, "# Next dump at t = %g, next snapshot at t = %g\n", t_next_dump, t_next_snapshot);
}

event defaults (i = 0) {
	if (is_constant (a.x))
		a = new face vector;

	foreach_face() {
		double f0 = 0.5*(f[] + f[-1]);
		sigma.x[] = sig(f0);
		epsilonA.x[] = epsA(f0);
	}
	boundary((scalar *){sigma, epsilonA});
}

// Compute the length of the jet
event front ( i += 1 ) {
	scalar m[];
	double THR = 1.e-2;
	foreach()
		m[] = (f[]>THR);
	tag(m);

	scalar c[];
	foreach()
		c[] = m[]==1 ? f[] : 0;
	scalar pos[];
	position (c, pos, {1., 0.});
	xFront = statsf(pos).max;
	if (xFront < 0 )
		xFront = 0.;
}

#if ELECTRIC
event electricAC ( i++ ) {
	foreach_face() {
		double f0 = 0.5*(f[] + f[-1]);
		sigma.x[] = sig(f0);
		epsilonA.x[] = epsA(f0);
		epsilonRWGF.x[] = epsilonA.x[]*fm.x[]/EPS0;
		alphaR.x[] = sigma.x[]*fm.x[];
		alphaI.x[] = 2.*PI*params.FREQ*epsilonA.x[]*fm.x[];
	}

	mgPhi = poisson_complex(phiR, phiI, alphaR, alphaI, tolerance = 1.e-8 );

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
}

event acceleration (i++) {
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

	if (t > params.TELEC) {
		face vector av = a;
		foreach_face()
			av.x[] += FORCE*(alpha.x[]/fm.x[]*(f_vec.x[] + f_vec.x[-1])/2.);
	}

	foreach()
		foreach_dimension() {
			FAC.x[] = f_vec.x[];
			FACFORCE.x[] = FORCE*(f_vec.x[]);
		}
}
#endif // ELECTRIC

event capillary_force (i++) {
	curvature (f, kappa);

	double sigma_val = f.sigma;
	foreach() {
		foreach_dimension() {
			double gradf = (f[1] - f[-1])/(2.*Delta);
			FC.x[] = sigma_val*kappa[]*gradf;
		}
	}
}

event logfile (i+=10) {
	if (i == 0)
		fprintf (stdout,
		         "# 1:t  2:dt  3:i  4:xFront  5:mgp.i  6:mgpf.i  7:mgu.i  8:mgPhi.i\n");
	fprintf (stdout, "%10.8e %6.4e %d %g %d %d %d %d\n",
	         t, dt, i, xFront, mgp.i, mgpf.i, mgu.i, mgPhi.i);
}

#if 1
event remove_bubbles_droplets ( i+=1 ) {
	xFront = remove_droplets_volmin (f, 0.01);
}
#endif

event end ( t = params.TEND ) {
	// Final dump before ending
	char nameDump[200];
	sprintf(nameDump, "dump-%014.4f", t);
	dump ( file = nameDump);
	dump ( file = "dump");

	// Update DUMP_FILE and save parameters
	strncpy(params.DUMP_FILE, nameDump, sizeof(params.DUMP_FILE) - 1);
	params.DUMP_FILE[sizeof(params.DUMP_FILE) - 1] = '\0';
	save_params("params.txt", &params);
	fprintf(stdout, "# Simulation ended at t = %g\n", t);
}

event dumps( i++ )
{
	if (t < t_next_dump)
		return 0;  // Not time yet

	if (t > params.TEND)
		return 0;  // Past end time

	char nameDump[200];
	sprintf(nameDump, "dump-%014.4f", t);
	dump ( file = nameDump);
	dump ( file = "dump");

	// Update DUMP_FILE to the latest dump and save parameters
	strncpy(params.DUMP_FILE, nameDump, sizeof(params.DUMP_FILE) - 1);
	params.DUMP_FILE[sizeof(params.DUMP_FILE) - 1] = '\0';
	save_params("params.txt", &params);

	// Schedule next dump
	t_next_dump += params.DTDUMP;
}

event postprocessing ( t = end )
{
	char label[200];
	view( tx = -1.5, ty = -0.3, sx = 2.0, sy = 12.0, width = 1200, height = 300 );

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

	// =========================================================
	// Compute and save jet radius along the longitudinal axis
	// =========================================================

	// Use output_facets to get interface coordinates (MPI-aware)
	// This outputs (x, y) pairs of the reconstructed interface segments
	char fname[200];
	sprintf(fname, "jet_radius-%g.dat", t);

	if (pid() == 0) {
		FILE *fp = fopen(fname, "w");
		fprintf(fp, "# Interface coordinates at t = %g\n", t);
		fprintf(fp, "# Column 1: x (longitudinal position)\n");
		fprintf(fp, "# Column 2: y (radial position = jet radius)\n");
		fclose(fp);
	}

	// output_facets appends interface segment coordinates
	// Each segment is output as two points separated by a blank line
	FILE *fp = fopen(fname, "a");
	output_facets(f, fp);
	fclose(fp);

	if (pid() == 0)
		fprintf(stdout, "# Jet radius (interface) saved to %s\n", fname);


	compute_electric_field();
	compute_electric_charge(EfR, EfI, qR, qI);
	compute_electric_force();

	// qR
	clear();
	box();
	sprintf(label, "qR t:%2.1f", t);
	squares("qR", cbar = true, pos = {0.5, -0.2}, levels = 10, mid = true, format = "%6.4f", label = label);
	draw_vof("f", lw = 2 , lc={0,0,0});
	save("qR.png");

	// qI
	clear();
	box();
	sprintf(label, "qI t:%2.1f", t);
	squares("qI", cbar = true, pos = {0.5, -0.2}, levels = 10, mid = true, format = "%6.4f", label = label);
	draw_vof("f", lw = 2 , lc={0,0,0});
	save("qI.png");

	// =========================================================
	// Save radial profiles of physical quantities
	// 5 radial lines per box, formatted for gnuplot
	// Each process handles its own box (pid() corresponds to box index)
	// =========================================================

	int nlines_per_box = 3;

	// Each process handles its own box
	// Process pid() owns x in [pid()*LBOX, (pid()+1)*LBOX]
	double x_min_local = pid() * params.LBOX;
	double dx_line = params.LBOX / nlines_per_box;

	for (int line = 0; line < nlines_per_box; line++) {
		// x position for this radial line (centered in each subdivision)
		double xpos = x_min_local + (line + 0.5) * dx_line;

		// Create filename for this radial profile (include pid to avoid conflicts)
		char profname[200];
		sprintf(profname, "radial_p%d_x%.4f_t%g.dat", pid(), xpos, t);

		FILE *fprad = fopen(profname, "w");
		fprintf(fprad, "# Radial profile at x = %g, t = %g\n", xpos, t);
		fprintf(fprad, "# 1:y 2:f 3:ux 4:uy 5:p 6:phiR 7:phiI 8:kappa 9:FCx 10:FCy 11:FACFORCEx 12:FACFORCEy 13:ERx 14:ERy 15:EIx 16:EIy 17:qR 18:qI\n");

		// Use direct cell access (no MPI synchronization needed)
		// foreach(serial) iterates over local cells only
		foreach(serial) {
			if (fabs(x - xpos) < Delta/2.) {
				fprintf(fprad, "%g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g\n",
				        y, f[], u.x[], u.y[], p[],
				        phiR[], phiI[], kappa[],
				        FC.x[], FC.y[], FACFORCE.x[], FACFORCE.y[],
				        ER.x[], ER.y[], EI.x[], EI.y[],
				        qR[], qI[]);
			}
		}

		fclose(fprad);
	}

	if (pid() == 0)
		fprintf(stdout, "# Radial profiles saved (%d files total)\n", NX * nlines_per_box);

	// Schedule next snapshot
	t_next_snapshot += params.DTOUT;
}
