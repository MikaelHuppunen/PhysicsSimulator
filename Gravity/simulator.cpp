#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <random>
#include <chrono>
#include <fstream>

double distance(std::vector<double>& coordinate1, std::vector<double>& coordinate2){
    return sqrt(pow(coordinate1[0]-coordinate2[0], 2)+pow(coordinate1[1]-coordinate2[1], 2)+pow(coordinate1[2]-coordinate2[2], 2));
}

double sign(double number){
    return (number > 0.0 ? 1.0 : -1.0);
}

double norm(std::vector<double> vector){
    return sqrt(pow(vector[0], 2)+pow(vector[1], 2)+pow(vector[2], 2));
}

double dot_product(std::vector<double>& vector1, std::vector<double>& vector2){
    return vector1[0]*vector2[0]+vector1[1]*vector2[1]+vector1[2]*vector2[2];
}

std::vector<double> vector_subtract(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vectors must be of the same length");
    }

    std::vector<double> result(vec1.size());
    for (std::size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] - vec2[i];
    }
    return result;
}

void write_memory_to_file(const std::string& filename, const std::vector<std::vector<std::vector<double>>>& memory) {
    // Prepend ./folder to the filename
    std::string filepath = "./data/" + filename;

    // Open the file
    std::ofstream outfile(filepath);

    // Check if the file was successfully opened
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open the file " << filepath << std::endl;
        return;
    }

    // Write the 3D vector to the file
    outfile << '[';
    for (std::size_t i = 0; i < memory.size(); ++i) {
        outfile << '[';
        for (std::size_t j = 0; j < memory[i].size(); ++j) {
            outfile << '[';
            for (std::size_t k = 0; k < memory[i][j].size(); ++k) {
                outfile << memory[i][j][k];
                if (k < memory[i][j].size() - 1) {
                    outfile << ',';
                }
            }
            outfile << ']';
            if (j < memory[i].size() - 1) {
                outfile << ',';
            }
        }
        outfile << ']';
        if (i < memory.size() - 1) {
            outfile << ',';
        }
    }
    outfile << ']' << '\n';

    // Close the file
    outfile.close();
}

std::vector<double> gravity(int index, double gravitational_constant, std::vector<double>& mass, std::vector<std::vector<double>>& position, std::vector<double>& radius){
    std::vector<double> acceleration = {0.0,0.0,0.0};
    for(int i = 0; i < mass.size(); i++){
        if(i != index){
            for(int axel = 0; axel < 3; axel++){
                //newton's law of gravity
                acceleration[axel] += -gravitational_constant*mass[i]/(pow(std::max(distance(position[index],position[i]), radius[index]+radius[i]), 3))\
                    *(position[index][axel]-position[i][axel]);
            }
        }
    }
    return acceleration;
}


void approximate(double delta_t, std::vector<double>& mass, std::vector<std::vector<double>>& velocity, std::vector<std::vector<double>>& position, std::vector<double>& radius, double gravitational_constant){
    for(int i = 0; i < mass.size(); i++){
        std::vector<double> acceleration = gravity(i, gravitational_constant, mass, position, radius);
        for(int axel = 0; axel < 3; axel++){
            velocity[i][axel] += delta_t*acceleration[axel];
            position[i][axel] += delta_t*velocity[i][axel];
        }
    }
}

double total_energy(std::vector<double>& mass, std::vector<std::vector<double>>& velocity, std::vector<std::vector<double>>& position, double gravitational_constant){
    double energy = 0;
    for(int i = 0; i < mass.size(); i++){
        energy += 0.5*mass[i]*(pow(velocity[i][0],2)+pow(velocity[i][1],2)+pow(velocity[i][2],2));
        for(int j = 0; j < mass.size()-i-1; j++){
            energy += -gravitational_constant*mass[i]*mass[i+j+1]/(distance(position[i],position[i+j+1]));
        }
    }
    return energy;
}

std::string HHMMSS(int seconds){
    int hours = seconds/3600;
    int minutes = (seconds % 3600)/60;
    seconds = seconds % 60;

    std::stringstream ss;
    ss << std::setw(2) << std::setfill('0') << hours << ":"
       << std::setw(2) << std::setfill('0') << minutes << ":"
       << std::setw(2) << std::setfill('0') << seconds;

    std::string formattedTime = ss.str();

    return formattedTime;
}

std::string time_left(int number_of_iterations, int number_of_simulation_iterations, double simulation_timer, int iteration, int simulation_iteration){
    double simulation_time_left = simulation_timer*(number_of_iterations*number_of_simulation_iterations-iteration*number_of_simulation_iterations-simulation_iteration-1.0+0.01)/(iteration*number_of_simulation_iterations+simulation_iteration+1.0+0.01);
    return HHMMSS(static_cast<int>(simulation_time_left));
}

class Space{
public:
    double gravitational_constant = 6.67384e-11;
    int action_size = 8;
    double scale = 5.0e8;
    double max_mass = 1.0e40;
    double max_distance = 1.0e20;
    double speed_of_light = 299792458.0;
    int dimensions = 2;
    double time_step = 600.0;
    int time_steps_per_step = 600;
    int datapoints_per_step = 10;

    Space(){
    }
    
    std::vector<double> get_initial_mass(){
        std::vector<double> mass = {1.9891e30, 5.9e24};
        return mass;
    }
    
    std::vector<std::vector<double>> get_initial_position(double angle, double multiplier){
        std::vector<std::vector<double>> position = {{-multiplier*cos(angle)*4.51e5,-multiplier*sin(angle)*4.51e5,0.0}};
        position.push_back({multiplier*cos(angle)*1.5210e11,multiplier*sin(angle)*1.5210e11,0.0});
        return position;
    }
    
    std::vector<std::vector<double>> get_initial_velocity(double angle, double multiplier){
        std::vector<std::vector<double>> velocity = {{-multiplier*cos(angle)*8.69e-2,-multiplier*sin(angle)*8.69e-2,0.0}};
        velocity.push_back({multiplier*cos(angle)*2.929e4,multiplier*sin(angle)*2.929e4,0.0});
        return velocity;
    }
    
    std::vector<double> get_initial_radius(){   
        std::vector<double> radius = {6.957e8,6.372e6};
        return radius;
    }
    
    
    std::vector<double> normalize_distance(std::vector<std::vector<double>>& position){
        std::vector<double> normalized_distance(position.size());
        std::vector<double> origin = {0.0,0.0,0.0};
        for(int i = 0; i < position.size(); i++){
            normalized_distance[i] = distance(origin, position[i])/scale;
        }
        return normalized_distance;
    }

    std::vector<double> get_angles(std::vector<std::vector<double>>& position){
        std::vector<double> angles(position.size());
        for(int i = 0; i < position.size(); i++){
            angles[i] = std::atan2(position[i][1], sign(position[i][0])*std::max(std::abs(position[i][0]) + 1.0, 1.0));
        }
        return angles;
    }
    
    std::vector<double> get_angular_velocity(std::vector<std::vector<double>>& position, std::vector<std::vector<double>>& velocity){
        std::vector<double> angular_velocities(position.size());
        for(int i = 0; i < position.size(); i++){
            angular_velocities[i] = time_step*time_steps_per_step*(position[i][0]*velocity[i][1]-position[i][1]*velocity[i][0])/(pow(norm(position[i]),2));
        }
        return angular_velocities;
    }
    
    std::vector<double> get_radial_velocity(std::vector<std::vector<double>>& position, std::vector<std::vector<double>>& velocity){
        std::vector<double> radial_velocities(position.size());
        for(int i = 0; i < position.size(); i++){
            radial_velocities[i] = time_step*time_steps_per_step*dot_product(velocity[i], position[i])/(scale*norm(position[i]));
        }
        return radial_velocities;
    }
    
    std::vector<std::vector<double>> get_acceleration(std::vector<double>& mass, std::vector<std::vector<double>>& position, std::vector<double>& radius){
        std::vector<std::vector<double>> acceleration(position.size());
        for(int i = 0; i < position.size(); i++){
            acceleration[i] = {gravity(i, gravitational_constant, mass, position, radius)};
            for(int j = 0; j < 3; j++){
                acceleration[i][j] *= pow((time_step*time_steps_per_step),2);
            }
        }
        return acceleration;
    }
    
    std::vector<double> get_radial_acceleration(std::vector<std::vector<double>>& position, std::vector<std::vector<double>>& velocity, std::vector<std::vector<double>>& acceleration, std::vector<double>& radial_velocity){
        std::vector<double> radial_acceleration(position.size());
        for(int i = 0; i < position.size(); i++){
            radial_acceleration[i] = (dot_product(acceleration[i], position[i])-pow(scale*radial_velocity[i], 2)+pow(time_step*time_steps_per_step*norm(velocity[i]), 2))/(scale*norm(position[i]));
        }
        return radial_acceleration;
    }
    
    std::vector<double> get_angular_acceleration(std::vector<std::vector<double>>& position, std::vector<std::vector<double>>& acceleration, std::vector<double>& radial_velocity, std::vector<double>& angular_velocity){
        std::vector<double> angular_acceleration(position.size());
        for(int i = 0; i < position.size(); i++){
            angular_acceleration[i] = (position[i][0]*acceleration[i][1]-position[i][1]*acceleration[i][0])/(pow(norm(position[i]),2))-2*scale*radial_velocity[i]*angular_velocity[i]/norm(position[i]);
        }
        return angular_acceleration;
    }
    
    std::vector<double> get_encoded_state(std::vector<double>& mass, std::vector<std::vector<double>>& position, std::vector<std::vector<double>>& velocity, std::vector<double>& radius){
        std::vector<double> normalized_distance = normalize_distance(position);
        std::vector<double> angles = get_angles(position);
        std::vector<double> radial_velocity = get_radial_velocity(position, velocity);
        std::vector<double> angular_velocity = get_angular_velocity(position, velocity);
        std::vector<std::vector<double>> acceleration = get_acceleration(mass, position, radius);
        std::vector<double> radial_acceleration = get_radial_acceleration(position, velocity, acceleration, radial_velocity);
        std::vector<double> angular_acceleration = get_angular_acceleration(position, acceleration, radial_velocity, angular_velocity);

        std::vector<double> encoded_state;
        encoded_state.reserve(
            normalized_distance.size() + angles.size() + radial_velocity.size() + angular_velocity.size() + 
            radial_acceleration.size() + angular_acceleration.size()
        );

        encoded_state.insert(encoded_state.end(), normalized_distance.begin(), normalized_distance.end());
        encoded_state.insert(encoded_state.end(), angles.begin(), angles.end());
        encoded_state.insert(encoded_state.end(), radial_velocity.begin(), radial_velocity.end());
        encoded_state.insert(encoded_state.end(), angular_velocity.begin(), angular_velocity.end());
        encoded_state.insert(encoded_state.end(), radial_acceleration.begin(), radial_acceleration.end());
        encoded_state.insert(encoded_state.end(), angular_acceleration.begin(), angular_acceleration.end());
        return encoded_state;
    }
    
    void simulate_next_state(std::vector<double>& mass, std::vector<std::vector<double>>& velocity, std::vector<std::vector<double>>& position, std::vector<double>& radius, int number_of_time_steps=0){
        if(number_of_time_steps == 0){
            number_of_time_steps = time_steps_per_step;
        }
        for(int i = 0; i < number_of_time_steps; i++){
            approximate(time_step, mass, velocity, position, radius, gravitational_constant);
        }
    }
    
    std::vector<std::vector<std::vector<double>>> get_next_state(std::vector<std::vector<double>>& velocity, std::vector<std::vector<double>>& position, std::vector<double> action){
        std::vector<double> distance_action(action.begin(), action.begin() + 2);
        std::vector<double> angle_action(action.begin() + 2, action.begin() + 4);
        std::vector<double> radial_velocity_action(action.begin() + 4, action.begin() + 6);
        std::vector<double> angular_velocity_action(action.begin() + 6, action.begin() + 8);

        std::vector<double> angles = get_angles(position);
        std::vector<double> radial_velocities = get_radial_velocity(position, velocity);
        std::vector<double> angular_velocities = get_angular_velocity(position, velocity);
        for(int i = 0; i < position.size(); i++){
            double new_angle = angles[i]+angle_action[i];
            double radial_velocity = scale*(radial_velocities[i]+radial_velocity_action[i])/(time_step*time_steps_per_step);
            double angular_velocity = (angular_velocities[i]+angular_velocity_action[i])/(time_step*time_steps_per_step);
            
            double distance_to_origin = norm(position[i]);
            distance_to_origin += distance_action[i]*scale;

            position[i][0] = distance_to_origin*cos(new_angle);
            position[i][1] = distance_to_origin*sin(new_angle);
            velocity[i][0] = radial_velocity*cos(new_angle)-distance_to_origin*angular_velocity*sin(new_angle);
            velocity[i][1] = radial_velocity*sin(new_angle)+distance_to_origin*angular_velocity*cos(new_angle);
        }
        
        return {position, velocity};
    }
};
    
class GravityAI{
public:
    int max_time_steps;
    int number_of_iterations;
    int number_of_simulation_iterations;

    GravityAI(int max_time_steps_input, int number_of_iterations_input, int number_of_simulation_iterations_inputs){
        max_time_steps = max_time_steps_input;
        number_of_iterations = number_of_iterations_input;
        number_of_simulation_iterations = number_of_simulation_iterations_inputs;
    }

    std::vector<std::vector<std::vector<double>>> simulation(){
        Space system;

        std::vector<std::vector<std::vector<double>>> memory;
        //get inital state
        std::random_device rd; // Seed generator
        std::mt19937 gen(rd()); // Mersenne Twister generator
        
        // Define the range for the random angle
        std::uniform_real_distribution<> dis1(-M_PI, M_PI);
        std::uniform_real_distribution<> dis2(0.9, 1.1);
        std::uniform_real_distribution<> dis3(-(M_PI/16.0-0.1), (M_PI/16.0-0.1));

        // Generate a random angle
        double random_angle = dis1(gen);
        double random_distance_multiplier = dis2(gen);
        double random_velocity_multiplier = dis2(gen);

        std::vector<double> mass = system.get_initial_mass();
        std::vector<std::vector<double>> position = system.get_initial_position(random_angle, random_distance_multiplier);
        random_angle += M_PI/2+dis3(gen);
        std::vector<std::vector<double>> velocity = system.get_initial_velocity(random_angle, random_velocity_multiplier);
        std::vector<double> radius = system.get_initial_radius();

        int time_step_count = 0;
        
        std::vector<std::vector<double>> old_distances(system.datapoints_per_step);
        std::vector<std::vector<double>> old_angles(system.datapoints_per_step);
        std::vector<std::vector<double>> old_radial_velocity(system.datapoints_per_step);
        std::vector<std::vector<double>> old_angular_velocity(system.datapoints_per_step);
        std::vector<std::vector<double>> old_encoded_states(system.datapoints_per_step);

        while(true){
            old_distances[time_step_count%system.datapoints_per_step] = system.normalize_distance(position);
            old_angles[time_step_count%system.datapoints_per_step] = system.get_angles(position);
            old_radial_velocity[time_step_count%system.datapoints_per_step] = system.get_radial_velocity(position, velocity);
            old_angular_velocity[time_step_count%system.datapoints_per_step] = system.get_angular_velocity(position, velocity);
            old_encoded_states[time_step_count%system.datapoints_per_step] = system.get_encoded_state(mass, position, velocity, radius);
            
            system.simulate_next_state(mass, velocity, position, radius, static_cast<int>(system.time_steps_per_step/system.datapoints_per_step));

            time_step_count++;
            if(time_step_count >= system.datapoints_per_step){
                std::vector<double> angles = system.get_angles(position);
                std::vector<double> distances = system.normalize_distance(position);
                std::vector<double> radial_velocity = system.get_radial_velocity(position, velocity);
                std::vector<double> angular_velocity = system.get_angular_velocity(position, velocity);

                std::vector<double> distance_action = vector_subtract(distances, old_distances[time_step_count%system.datapoints_per_step]);
                std::vector<double> angle_action = vector_subtract(angles, old_angles[time_step_count%system.datapoints_per_step]);
                std::vector<double> radial_velocity_action = vector_subtract(radial_velocity, old_radial_velocity[time_step_count%system.datapoints_per_step]);
                std::vector<double> angular_velocity_action = vector_subtract(angular_velocity, old_angular_velocity[time_step_count%system.datapoints_per_step]);

                for(int i = 0; i < angle_action.size(); i++){
                    angle_action[i] = std::fmod(angle_action[i] + M_PI, 2.0 * M_PI);
                    if (angle_action[i] < 0) {
                        angle_action[i] += 2.0 * M_PI;
                    }
                    angle_action[i] -= M_PI;
                }

                std::vector<double> action;
                action.reserve(
                    distance_action.size() + angle_action.size() + radial_velocity_action.size() + angular_velocity_action.size()
                );

                action.insert(action.end(), distance_action.begin(), distance_action.end());
                action.insert(action.end(), angle_action.begin(), angle_action.end());
                action.insert(action.end(), radial_velocity_action.begin(), radial_velocity_action.end());
                action.insert(action.end(), angular_velocity_action.begin(), angular_velocity_action.end());
                
                memory.push_back({old_encoded_states[time_step_count%system.datapoints_per_step], action});
            }
            if(time_step_count/system.datapoints_per_step >= max_time_steps){
                return memory;
            }
        }
    }
    
    void learn(){
        std::vector<std::vector<std::vector<double>>> memory;
        double simulation_timer = 0.0;
        for(int iteration = 0; iteration < number_of_iterations; iteration++){
            for(int simulation_iteration = 0; simulation_iteration < number_of_simulation_iterations; simulation_iteration++){
                auto start = std::chrono::high_resolution_clock::now();
                std::vector<std::vector<std::vector<double>>> simualtion_memory = simulation();
                memory.insert(memory.end(), simualtion_memory.begin(), simualtion_memory.end());
                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                simulation_timer += duration.count()/1000000.0;
                if(simulation_iteration%10 == 0 || simulation_iteration == number_of_simulation_iterations-1){
                    std::cout << iteration+1 << '/' << number_of_iterations << ':' << 100.0*(static_cast<double>(simulation_iteration)+1.0)/static_cast<double>(number_of_simulation_iterations) << "%, estimated time left: " << time_left(number_of_iterations, number_of_simulation_iterations, simulation_timer, iteration, simulation_iteration) << 's' << '\n';
                }
            }
        }
        write_memory_to_file("simulation.json", memory);
    }
};

void learn(int max_time_steps, int number_of_iterations, int number_of_simulation_iterations){
    GravityAI gravityai(max_time_steps, number_of_iterations, number_of_simulation_iterations);
    auto start = std::chrono::high_resolution_clock::now();
    gravityai.learn();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "simulation time: " << HHMMSS(duration.count()/1000000.0) << "s\n";
}

int main(){
    int max_time_steps = 10;
    int number_of_iterations = 1;
    int number_of_simulation_iterations = 131072;
    learn(max_time_steps, number_of_iterations, number_of_simulation_iterations);
    return 0;
}