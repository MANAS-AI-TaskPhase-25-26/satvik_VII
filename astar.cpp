#include <rclcpp/rclcpp.hpp>                  // Core ROS 2 C++ library
#include <nav_msgs/msg/occupancy_grid.hpp>    // The message type for the 2D map
#include <nav_msgs/msg/path.hpp>              // The message type for the final route
#include <geometry_msgs/msg/pose_stamped.hpp> // The message type for individual points on the path
#include <vector>                             // Standard C++ dynamic arrays (lists)
#include <cmath>                              // Math functions (like square root)
#include <algorithm>                          // Algorithms (like reversing a list)

// ============================================================================
// 1. SIMPLE COORDINATE HOLDER
// ============================================================================
// point structure for frid coords
struct Point {
    int x;
    int y;
    
    // defining the meaning for == in case of point structure 
    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
};

// ============================================================================
// 2. THE MAIN ROS 2 NODE CLASS
// ============================================================================
// We inherit from rclcpp::Node, which gives this class all the powers of a ROS 2 node.
class AStarPlanner : public rclcpp::Node {
public:
    // CONSTRUCTOR: This runs exactly once when the node is created.
    AStarPlanner() : Node("astar_planner") {
        
        // --- MAP SUBSCRIBER SETUP ---
        // set to transient local so that even after the map is published we can recive it and use it 
        rclcpp::QoS map_qos(10);
        map_qos.transient_local();
        
        // We create the subscriber. It listens to the "/map" topic. 
        // Whenever a map arrives, it automatically triggers the "map_callback" function below.
        map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/map", map_qos, std::bind(&AStarPlanner::map_callback, this, std::placeholders::_1));
            
        // --- PATH PUBLISHER SETUP ---
        // We create a publisher that will broadcast our final calculated route on the "/path" topic.
        // The '10' is the queue size (it remembers up to 10 messages if the network is slow).
        path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/path", 10);
        
        // Print a message to the terminal so we know it's alive.
        RCLCPP_INFO(this->get_logger(), "A* Planner Node Started. Waiting for map...");
    }

private:
    // Pointers to hold our subscriber and publisher alive in memory
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;

    // ========================================================================
    // TRIGGERED WHEN THE MAP ARRIVES
    // ========================================================================
    // 'msg' contains all the data about the map (width, height, resolution, and the actual grid data).
    void map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(), "Map received! Computing 2D A* path...");
        
        // Extract width and height of the map 
        int width = msg->info.width;
        int height = msg->info.height;
        
        // Define where we want to start and end. 
        // Right now, this is hardcoded to go from the bottom-left to the top-right of the map.(creates problems for map3)
        // fix by adding paddint to top and bottom of map 3
        Point start = {0, 0};
        Point goal = {width - 1, height - 1};

        // Call A* function to do the math. 
        // It returns a list (vector) of Points representing the winning path.
        std::vector<Point> path = calculateAStar(start, goal, msg->data, width, height);

        // If the path list isn't empty, publish it.
        if (!path.empty()) {
            publish_path(path, msg);
        } else {
            // If it is empty, A* gave up. The goal is completely blocked off.
            RCLCPP_WARN(this->get_logger(), "Failed to find a valid path! The goal might be blocked.");
        }
    }

    // ========================================================================
    // HELPER: HEURISTIC (THE GUESSER)
    // ========================================================================
    // calculates the straight-line ("Euclidean") distance between two points.
    // used formula: sqrt( (x1-x2)^2 + (y1-y2)^2 )
    float heuristic(Point a, Point b) {
        return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
    }

    // ========================================================================
    // HELPER: COLLISION CHECKER
    // ========================================================================
    // This checks if a specific X, Y coordinate is safe for the robot to step on.
    bool isSafeToWalk(int x, int y, int width, int height, const std::vector<int8_t>& flat_map) {
        // 1. BOUNDARY CHECK: Are we trying to step off the edge of the map?
        if (x < 0 || x >= width || y < 0 || y >= height) return false;
        
        // 2. 2D TO 1D CONVERSION: 
        // ROS maps aren't sent as a 2D grid (like map[x][y]). They are sent as one long, flat list of numbers.
        // To find a 2D (x,y) point in a 1D list, we multiply the Y row by the total width, then add X.
        int index = y * width + x; 
        
        // 3. OBSTACLE CHECK: 
        // ROS map values range from 0 (free space) to 100 (solid wall). -1 means "unknown area".
        // Here, we say any cell with a value over 50 is too dangerous, and unknown areas (-1) are rejected.
        if (flat_map[index] > 50 || flat_map[index] == -1) return false;
        
        // If it passes all tests, it's a safe floor tile!
        return true; 
    }

    // ========================================================================
    // THE CORE A* ALGORITHM
    // ========================================================================
    std::vector<Point> calculateAStar(Point start, Point goal, const std::vector<int8_t>& flat_map, int width, int height) {
        
        // --- 1. SETUP TRACKING GRIDS ---
        // A* needs to remember scores for every single tile on the map.
        // We create 2D vectors matching the map size and fill them with 999999.0 to represent "Infinity".
        
        // g_cost: The EXACT distance we walked from the start point to get to this tile.
        std::vector<std::vector<float>> g_cost(width, std::vector<float>(height, 999999.0));
        
        // f_cost: The TOTAL estimated trip distance (g_cost + guessed remaining distance to goal).
        std::vector<std::vector<float>> f_cost(width, std::vector<float>(height, 999999.0));
        
        // came_from: Our breadcrumb trail. For every tile, it remembers the tile we stood on right before it.
        // We initialize it with a fake coordinate (-1, -1) so we know when to stop looking backward.
        std::vector<std::vector<Point>> came_from(width, std::vector<Point>(height, {-1, -1}));
        
        // closed_list: A map of "Tiles we have already fully explored and shouldn't look at again."
        std::vector<std::vector<bool>> closed_list(width, std::vector<bool>(height, false));

        // --- 2. SETUP THE "TO-DO" LIST (The Open Set) ---
        // This holds all the tiles we have discovered but haven't stepped on yet.
        std::vector<Point> to_do_list;

        // Initialize the starting point! 
        // Distance from start to start is 0.
        g_cost[start.x][start.y] = 0.0;
        // Total trip estimate is just the straight line guess to the goal.
        f_cost[start.x][start.y] = heuristic(start, goal);
        // Add the start point to our to-do list so the loop has somewhere to begin.
        to_do_list.push_back(start);

        // --- MOVEMENT RULES ---
        // These two arrays define the 8 directions we can step: 
        // Left, Right, Down, Up (the first 4)
        // Down-Left, Up-Left, Down-Right, Up-Right (the last 4 diagonals)
        int move_x[] = {-1, 1, 0, 0, -1, -1, 1, 1};
        int move_y[] = {0, 0, -1, 1, -1, 1, -1, 1};

        // --- 3. MAIN LOOP ---
        // Keep exploring as long as we have tiles on our to-do list.
        while (to_do_list.size() > 0) {

            // ========================================================================
            // Step 3a: FIND THE MOST PROMISING TILE
            // ========================================================================

            // 1. Set a fake "record" F-cost that is impossibly high.
            // We do this so the very first real tile we look at is guaranteed to beat it.
            float lowest_f = 999999.0;

            // 2. Create a variable to remember the exact slot number (index) 
            // of the winning tile in our to-do list.
            int best_index = 0;

            // 3. Loop through every single tile currently sitting on our to-do list.
            // 'i' is the current slot number we are looking at.
            for (size_t i = 0; i < to_do_list.size(); i++) {
                
                // Grab the actual X,Y coordinate out of slot 'i'
                Point p = to_do_list[i];
                
                // Check our giant f_cost grid: Is the score for this specific X,Y tile
                // LOWER than our current standing record?
                if (f_cost[p.x][p.y] < lowest_f) {
                    
                    // WE HAVE A NEW WINNER! 
                    // Update the record to beat...
                    lowest_f = f_cost[p.x][p.y];
                    
                    // ...and remember the exact slot number this winning tile is sitting in.
                    best_index = i;
                }
            }
            // When this loop finishes, 'best_index' will hold the slot number of the best possible move.


            // Grab the best point, and REMOVE it from the to-do list.
            Point current = to_do_list[best_index];
            to_do_list.erase(to_do_list.begin() + best_index);
            
            // Mark this tile as fully explored. We won't process it again.
            closed_list[current.x][current.y] = true;

            // Step 3b: DID WE REACH THE GOAL?
            if (current == goal) {
                std::vector<Point> final_path;
                Point step = goal;
                
                // We won! Now we follow the breadcrumbs backwards.
                // We keep asking the 'came_from' map: "How did I get here?" until we hit our fake (-1,-1) start.
                while (!(step.x == -1 && step.y == -1)) { 
                    final_path.push_back(step);          // Add the step to the path
                    step = came_from[step.x][step.y];    // Jump to the parent step
                }
                
                // Because we tracked it backwards (Goal to Start), we need to reverse the list so the robot 
                // gets the instructions forwards (Start to Goal).
                std::reverse(final_path.begin(), final_path.end());
                return final_path; // WE ARE DONE! Exit the function and return the path.
            }

            // Step 3c: CHECK ALL 8 NEIGHBORS of our current tile.
            for (int i = 0; i < 8; i++) {
                // Calculate the exact X,Y of the neighbor we are looking at.
                Point neighbor = {current.x + move_x[i], current.y + move_y[i]};

                // Skip this neighbor if it's a wall or off the map.
                if (!isSafeToWalk(neighbor.x, neighbor.y, width, height, flat_map)) continue;
                
                // Skip this neighbor if we've already fully explored it.
                if (closed_list[neighbor.x][neighbor.y]) continue; 

                // How much does it cost to take this step? 
                // If it's one of the first 4 directions (straight), cost is 1.0.
                // If it's a diagonal, math says the distance is square root of 2 (approx 1.414).
                float step_cost = (i < 4) ? 1.0 : 1.414;
                
                // Calculate the "tentative" distance from Start to this Neighbor going through our Current tile.
                float tentative_g = g_cost[current.x][current.y] + step_cost;

                // Step 3d: IS THIS A RECORD BREAKING ROUTE?
                // If our new tentative distance is LOWER than whatever distance is currently recorded 
                // for that neighbor, it means we found a faster way to get to that tile!
                if (tentative_g < g_cost[neighbor.x][neighbor.y]) {
                    
                    // Update the breadcrumb so the neighbor points back to 'current' as its best parent.
                    came_from[neighbor.x][neighbor.y] = current;
                    
                    // Update its G cost (distance from start) and F cost (total trip estimate)
                    g_cost[neighbor.x][neighbor.y] = tentative_g;
                    f_cost[neighbor.x][neighbor.y] = tentative_g + heuristic(neighbor, goal);
                    
                    // Finally, add this neighbor to the to-do list so we can eventually stand on it 
                    // and look at ITS neighbors. (We do a quick check to make sure we don't add duplicates).
                    bool already_in_list = false;
                    for (Point p : to_do_list) {
                        if (p == neighbor) already_in_list = true;
                    }
                    if (!already_in_list) to_do_list.push_back(neighbor);
                }
            }
        }
        
        // If the while loop runs out of things to do and never hits the "current == goal" check,
        // it means we explored every reachable tile and couldn't find a way. Return an empty path.
        return {}; 
    }

    // ========================================================================
    // CONVERT AND PUBLISH
    // ========================================================================
    // A* gave us a list of "Grid Coordinates" (like cell x:5, y:10).
    // Robots don't know what cells are. They need "World Coordinates" in meters.
    void publish_path(const std::vector<Point>& grid_path, const nav_msgs::msg::OccupancyGrid::SharedPtr& map_msg) {
        
        nav_msgs::msg::Path path_msg;
        path_msg.header.stamp = this->get_clock()->now(); // Timestamp it
        path_msg.header.frame_id = "map";                 // Tell ROS this path exists in the "map" coordinate frame
        
        // 'resolution' is how big one cell is in meters (e.g., 0.05 meters per cell).
        float res = map_msg->info.resolution;
        
        // 'origin' is where cell (0,0) actually sits in the real world. 
        // It's usually a negative number to put the center of the map at 0,0.
        float origin_x = map_msg->info.origin.position.x;
        float origin_y = map_msg->info.origin.position.y;

        // Loop through every grid point in our calculated path
        for (const auto& p : grid_path) {
            geometry_msgs::msg::PoseStamped pose;
            pose.header = path_msg.header;
            
            // THE MATH: World_Pos = Map_Origin + (Grid_Index * Resolution)
            // We also add (res / 2.0). Why? Because multiplying by resolution puts the dot exactly 
            // on the bottom-left corner of the grid square. Adding half a resolution shifts the dot 
            // to the exact dead-center of the square, which keeps the robot away from the walls!
            pose.pose.position.x = origin_x + (p.x * res) + (res / 2.0);
            pose.pose.position.y = origin_y + (p.y * res) + (res / 2.0);
            pose.pose.position.z = 0.0; // Path is flat on the ground
            
            // A quaternion representing no rotation. We just want points, not facing angles.
            pose.pose.orientation.w = 1.0; 
            
            // Add the converted meter point to our ROS message list
            path_msg.poses.push_back(pose);
        }

        // Send it out to the ROS network!
        path_pub_->publish(path_msg);
        RCLCPP_INFO(this->get_logger(), "Path published successfully!");
    }
};

// ============================================================================
// C++ MAIN FUNCTION
// ============================================================================
int main(int argc, char **argv) {
    // 1. Initialize ROS 2
    rclcpp::init(argc, argv);
    
    // 2. Create our Node and 'spin' it. 
    // "Spinning" tells ROS to keep the program open and constantly check for new map messages.
    rclcpp::spin(std::make_shared<AStarPlanner>());
    
    // 3. Clean up when the user hits Ctrl+C
    rclcpp::shutdown();
    return 0;
}
