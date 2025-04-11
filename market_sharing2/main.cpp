#include <ilcplex/ilocplex.h>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <numeric> 
#include <algorithm> 

ILOSTLBEGIN

int main() {
    IloEnv env;
    IloModel model(env);

    const int numRetailers = 23;
    std::vector<int> retailers_ids(numRetailers);
    for (int i = 0; i < numRetailers; ++i) retailers_ids[i] = i;

    std::map<int, int> region;
    std::map<int, int> deliv;
    std::map<int, double> oil_m;
    std::map<int, double> spi_m; 
    std::map<int, char> growth;

    std::vector<int> region1_indices, region2_indices, region3_indices;
    std::vector<int> group_a_indices, group_b_indices;

    std::vector<int> deliv_values = { 11, 47, 44, 25, 10, 26, 26, 54, 18, 51, 20, 105, 7, 16, 34, 100, 50, 21, 11, 19, 14, 10, 11 };
    std::vector<double> oil_values = { 9, 13, 14, 17, 18, 19, 23, 21, 9, 11, 17, 18, 18, 17, 22, 24, 36, 43, 6, 15, 15, 25, 39 };
    std::vector<double> spi_values = { 34, 411, 82, 157, 5, 183, 14, 215, 102, 21, 54, 0, 6, 96, 118, 112, 535, 8, 53, 28, 69, 65, 27 };
    std::vector<char> growth_values = { 'A', 'A', 'A', 'B', 'A', 'A', 'B', 'B', 'B', 'A', 'B', 'B', 'B', 'B', 'A', 'B', 'B', 'B', 'B', 'A', 'B', 'B', 'B' };

    for (int i = 0; i < numRetailers; ++i) {
        deliv[i] = deliv_values[i];
        oil_m[i] = oil_values[i];
        spi_m[i] = spi_values[i];
        growth[i] = growth_values[i];

        if (i <= 7) { // Retailers M1-M8
            region[i] = 1;
            region1_indices.push_back(i);
        }
        else if (i <= 17) { // Retailers M9-M18
            region[i] = 2;
            region2_indices.push_back(i);
        }
        else { // Retailers M19-M23
            region[i] = 3;
            region3_indices.push_back(i);
        }

        if (growth_values[i] == 'A') {
            group_a_indices.push_back(i);
        }
        else {
            group_b_indices.push_back(i);
        }
    }

    double total_deliv = std::accumulate(deliv_values.begin(), deliv_values.end(), 0.0);
    double total_spi = std::accumulate(spi_values.begin(), spi_values.end(), 0.0);
    double total_oil_r1 = 0, total_oil_r2 = 0, total_oil_r3 = 0;
    for (int i : region1_indices) total_oil_r1 += oil_m[i];
    for (int i : region2_indices) total_oil_r2 += oil_m[i];
    for (int i : region3_indices) total_oil_r3 += oil_m[i];
    double total_group_a = group_a_indices.size();
    double total_group_b = group_b_indices.size();

    std::vector<double> totals = { total_oil_r1, total_oil_r2, total_oil_r3, total_group_a, total_group_b, total_deliv, total_spi };
    std::vector<double> targets(7);
    std::vector<double> deviation_bounds(7);
    for (int k = 0; k < 7; ++k) {
        targets[k] = totals[k] * 0.40;
        deviation_bounds[k] = totals[k] * 0.05;
    }


    IloBoolVarArray x(env, numRetailers);
    for (int i = 0; i < numRetailers; ++i) {
        x[i].setName(("x_" + std::to_string(i + 1)).c_str());
    }

    IloNumVarArray sl(env, 7, 0.0, IloInfinity);
    IloNumVarArray su(env, 7, 0.0, IloInfinity);

    for (int k = 0; k < 7; ++k) {
        sl[k].setUB(deviation_bounds[k] + 1e-9);
        su[k].setUB(deviation_bounds[k] + 1e-9);
        sl[k].setName(("sl_" + std::to_string(k)).c_str());
        su[k].setName(("su_" + std::to_string(k)).c_str());
    }


    IloExpr objective(env);
    for (int k = 0; k < 7; ++k) {
        if (targets[k] > 1e-9) {
            objective += (sl[k] + su[k]) / targets[k];
        }
        else {
            objective += (sl[k] + su[k]) * 1e6;
        }
    }
    model.add(IloMinimize(env, objective));
    objective.end();

    // 1. Oil Market Region 1 (Goal k=0)
    IloExpr share_oil_r1(env);
    for (int i : region1_indices) share_oil_r1 += oil_m[i] * x[i];
    model.add(share_oil_r1 + sl[0] - su[0] == targets[0]);
    share_oil_r1.end();

    // 2. Oil Market Region 2 (Goal k=1)
    IloExpr share_oil_r2(env);
    for (int i : region2_indices) share_oil_r2 += oil_m[i] * x[i];
    model.add(share_oil_r2 + sl[1] - su[1] == targets[1]);
    share_oil_r2.end();

    // 3. Oil Market Region 3 (Goal k=2)
    IloExpr share_oil_r3(env);
    for (int i : region3_indices) share_oil_r3 += oil_m[i] * x[i];
    model.add(share_oil_r3 + sl[2] - su[2] == targets[2]);
    share_oil_r3.end();

    // 4. Number of Group A Retailers (Goal k=3)
    IloExpr share_group_a(env);
    for (int i : group_a_indices) share_group_a += x[i];
    model.add(share_group_a + sl[3] - su[3] == targets[3]);
    share_group_a.end();

    // 5. Number of Group B Retailers (Goal k=4)
    IloExpr share_group_b(env);
    for (int i : group_b_indices) share_group_b += x[i];
    model.add(share_group_b + sl[4] - su[4] == targets[4]);
    share_group_b.end();

    // 6. Total Delivery Points (Goal k=5)
    IloExpr share_deliv(env);
    for (int i = 0; i < numRetailers; ++i) share_deliv += deliv[i] * x[i];
    model.add(share_deliv + sl[5] - su[5] == targets[5]);
    share_deliv.end();

    // 7. Total Spirit Market (Goal k=6)
    IloExpr share_spi(env);
    for (int i = 0; i < numRetailers; ++i) share_spi += spi_m[i] * x[i];
    model.add(share_spi + sl[6] - su[6] == targets[6]);
    share_spi.end();

    IloCplex cplex(model);
    cplex.exportModel("market_sharing_corrected_v2.lp");
    cplex.solve();

    env.out() << "Solution status = " << cplex.getStatus() << endl;
    env.out() << "Solution value  = " << cplex.getObjValue() << endl;

    env.out() << "\nRetailers in D1 (Optimal Solution):" << endl;
    std::vector<int> d1_retailers;
    for (int i = 0; i < numRetailers; ++i) {
        if (cplex.getValue(x[i]) > 0.9) {
            d1_retailers.push_back(i + 1);
            env.out() << "M" << (i + 1) << " ";
        }
    }
    env.out() << endl;

    env.out() << "\nRetailers in D2:" << endl;
    std::vector<int> d2_retailers;
    for (int i = 0; i < numRetailers; ++i) {
        if (cplex.getValue(x[i]) < 0.1) {
            d2_retailers.push_back(i + 1);
            env.out() << "M" << (i + 1) << " ";
        }
    }
    env.out() << endl;

    env.out() << "\nDeviations from 40% target:" << endl;
    std::vector<std::string> goal_names = { "Oil R1", "Oil R2", "Oil R3", "Group A", "Group B", "Delivery", "Spirit" };
    for (int k = 0; k < 7; ++k) {
        double achieved_val = targets[k] - cplex.getValue(sl[k]) + cplex.getValue(su[k]);
        double achieved_pct = (totals[k] > 1e-9) ? (achieved_val / totals[k] * 100.0) : 0.0;
        // Calculate proportional deviation sum for objective comparison
        double obj_term_dev = (targets[k] > 1e-9) ? ((cplex.getValue(sl[k]) + cplex.getValue(su[k])) / targets[k]) : ((cplex.getValue(sl[k]) + cplex.getValue(su[k])) * 1e6);

        env.out() << "Goal " << k << " (" << goal_names[k] << "): Target=" << targets[k]
            << ", Achieved=" << achieved_val << " (" << achieved_pct << "%)"
                << ", Sl=" << cplex.getValue(sl[k]) << ", Su=" << cplex.getValue(su[k])
                << ", Obj Term=" << obj_term_dev << endl;
    }

    env.end();
    return 0;
}