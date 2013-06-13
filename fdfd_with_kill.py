""" If something goes wrong, issues 'halt stop'.

    Useful in AWS, where we would like to kill an abnormal node.
"""

import sys
import subprocess # For killing the node.

DIAGNOSTIC_SIMULATION = "/home/fdfd/test/200x200x200"

def run_fdfd(name):
    
    # Make sure we actually have the expected number of GPUs.
    try:
        import fdfd
    except TypeError: # Means we could not find the required GPUs.
        kill_instance()


    # Run the simulation.
    success = fdfd.simulate(name)

    # If simulation did not succeed, run diagnostic simulation.
    if not success:
        print "Did not terminate properly, running diagnostic simulation."
        diag_success = fdfd.simulate(DIAGNOSTIC_SIMULATION, check_success_only=True)
        if not diag_success: # Something's wrong with the node.
            print "Diagnostic simulation failed!"
            kill_instance()
        else:
            print "Diagnostic simulation returned success. Node checks out."


def kill_instance():
    print "Attempting instance shutdown!"
    subprocess.call(["sudo", "halt"])


if __name__ == '__main__': # Allows calls from command line.
    run_fdfd(sys.argv[1]) # Specify name of the job.

