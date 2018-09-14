/*
 * A miniapp that demonstrates using an external spike source.
 */

#include <fstream>
#include <iomanip>
#include <iostream>

#include <nlohmann/json.hpp>

#include <arbor/assert_macro.hpp>
#include <arbor/common_types.hpp>
#include <arbor/context.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/mc_cell.hpp>
#include <arbor/profile/meter_manager.hpp>
#include <arbor/profile/profiler.hpp>
#include <arbor/simple_sampler.hpp>
#include <arbor/simulation.hpp>
#include <arbor/recipe.hpp>
#include <arbor/version.hpp>

#include <aux/ioutil.hpp>
#include <aux/json_meter.hpp>
#include <aux/with_mpi.hpp>
#include <mpi.h>

#include "mpiutil.hpp"

using arb::cell_gid_type;
using arb::cell_lid_type;
using arb::cell_size_type;
using arb::cell_member_type;
using arb::cell_kind;
using arb::time_type;

//
//  N ranks = Nn + Na
//      Nn = number of nest ranks
//      Na = number of arbor ranks
//
//  Nest  on COMM_WORLD [0, Nn)
//  Arbor on COMM_WORLD [Nn, N)
//

int main(int argc, char** argv) {
    try {
        aux::with_mpi guard(argc, argv, false);

        //  INITIALISE MPI

        auto info = get_comm_info(false);

        //  MODEL SETUP

        int num_nest_cells = 10;


        //  HAND SHAKE ARBOR-NEST

        // Get simulation length from Arbor
        float sim_duration = broadcast(0.f, MPI_COMM_WORLD, info.arbor_root);
        int num_arbor_cells = broadcast(0,   MPI_COMM_WORLD, info.arbor_root);
        broadcast(num_nest_cells,   MPI_COMM_WORLD, info.nest_root);

        float min_delay    = broadcast(0.f, MPI_COMM_WORLD, info.arbor_root);

        float delta = min_delay/2;
        unsigned steps = sim_duration/delta;
        if (steps*delta<sim_duration) ++steps;

        int  total_cells = num_arbor_cells + num_nest_cells;

        //  BUILD NEST PROXY MODEL
        std::vector<int> local_cells;
        for (int gid=num_arbor_cells+info.local_rank; gid<total_cells; gid+=info.nest_size) {
            local_cells.push_back(gid);
        }
        print_vec_comm(local_cells, info.comm);

        //  SEND SPIKES TO ARBOR (RUN SIMULATION)

        for (unsigned step=0; step<=steps; ++step) {
            //std::cout << "NEST: callback " << step << " at t " << step*delta << std::endl;

            std::vector<arb::spike> local_spikes;
            if (!step) {
                for (unsigned gid: local_cells) {
                    arb::spike s;
                    s.source = {gid, 0u};
                    s.time = (float)(gid-num_arbor_cells);
                    local_spikes.push_back(s);
                }
                print_vec_comm(local_spikes, info.comm);
            }
            auto v = gather_spikes(local_spikes, MPI_COMM_WORLD);
            if (v.size()) print_vec_comm(v, info.comm);

        }
    }
    catch (std::exception& e) {
        std::cerr << "exception caught in ring miniapp:\n" << e.what() << "\n";
        return 1;
    }

    return 0;
}