#!/usr/bin/env perl
# This script is modified by Xiaorui Dong according to https://github.com/aspuru-guzik-group/xtb-gaussian/blob/master/xtb-gaussian
# Changes: - allow dynamic assignment of uhf;

# An interface between Gaussian and xtb that takes care of Hessian
# calculations and command line arguments.
use strict;
use warnings;
use File::Basename;

# if the first argument is --log-all, then the full xtb output will be
# added to the Gaussian log file
my $DEBUG = 0;
if ($ARGV[0] eq '--log-all') {
    $DEBUG = 1;
    shift;
}

# Final 6 arguments are those passed by Gaussian.
my @arg_gauss = splice @ARGV, @ARGV - 6;

# Remaining arguments are for xtb.
my $arg_xtb = join ' ', @ARGV;

# First, move to the directory containing the .EIn file (the Gaussian scratch
# directory.) This is so that xtb can produce a .EOut file in the same
# directory.
(my $Ein,  my $Ein_dir)  = fileparse($arg_gauss[1]);
(my $Eout, my $Eout_dir) = fileparse($arg_gauss[2]);
(my $Elog, my $Elog_dir) = fileparse($arg_gauss[3]);

chdir $Ein_dir;
# Open input file and load parameters.
open (INF, "<$Ein") || die "failed to open input";
my ($natoms,$deriv,$icharg,$multip) = split(" ",scalar(<INF>));
close(INF);

# Setup redirection of xtb output. Here we throw it out instead, unless $DEBUG
# is on. We do this because otherwise the Gaussian output gets way too
# cluttered.
my $msg_output =  $DEBUG ? "> $Elog 2>&1"  :  ">/dev/null 2>$Elog";

# Setup xtb according to run type
my $runtype = $deriv<2 ? "--grad" : "--hess";
# Modified by Xiaorui
my $uhf = $multip - 1;
my $xtb_run = "xtb ./$Ein $arg_xtb $runtype --charge $icharg --uhf $uhf $msg_output";
system($xtb_run);

open(LOGF, ">>$Elog") || die "couldn't open log file";
print LOGF "\n------- xtb command was ---------\n";
print LOGF "?> $xtb_run\n";
print LOGF "---------------------------------\n";

# Modified by Xiaorui
# Currently, non-singlet spins are not supported explicitly by the interface.
# unless ($multip == 1) {
#     print LOGF "WARNING: Gaussian multiplicity S=$multip is not singlet.\n";
#     print LOGF "         This is not not explicitly supported. Results are likely wrong without\n";
#     print LOGF "         an appropriate --uhf argument xtb command line!\n";}


if ($deriv < 2) {
    # This is the standard and identical to the xtb example from the repo. If
    # we don't need a hessian, just run xtb with the correct Gaussian
    # input/output file formats and --grad.

} else {
    # Appending to xtb gaussian formatted output
    open(OUTP, ">>$Eout") || die "failed to open output";

    # First, we fake the polarizability and the dipole derivatives, which the
    # Gaussian Manual says should be of this form,

    # Polar(I), I=1,6          3D20.12
    # DDip(I), I=1,9*NAtoms    3D20.12
    foreach (1 .. 3*$natoms + 2) {
        printf OUTP "%20.12e%20.12e%20.12e\n", 0.0, 0.0, 0.0;
    }

    # Now we have to convert the hessian from the turbomole format that xtb
    # outputs to the Gaussian format that we want. This is fairly trivial,
    open(HESSF, "<hessian") || die "failed to open hessian";

    # We don't need the first line that just contains $hessian
    <HESSF>;

    # Now we are iterating over Hessian matrix elements. We will
    # append those to the output file in the correct format, given in
    # the Gaussian manual as

    # FFX(I), I=1,(3*NAtoms*(3*NAtoms+1))/2      3D20.12

    # That is, the lower triangular part of the Hessian only. For this we need
    # to remember which indices we have done.
    my $icol=0; my $irow=0;
    # Finally we only print three numbers per line,
    my $counter3=0;

    while (<HESSF>) {
        # split the line on whitespaces
        foreach (split){


            # print only if the column index < row index (lower triangle)
            if ($icol <= $irow) {
                # If printed more than three in a row, start new line
                if ($counter3 == 3) {$counter3=0; print OUTP "\n";}
                printf OUTP "%20.12e",  $_;
                # print OUTP "$icol x $irow ($_)   ";
                $counter3++;
            }

            # Increment column index
            $icol++;
            if ($icol == (3*$natoms)) { # done this row
                $irow++; $icol = 0;
            }
        }
    }

    # Close and flush
    close(OUTP);
}

# Close log and flush
print LOGF "             Control returned to Gaussian.\n";
print LOGF "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
close(LOGF);
