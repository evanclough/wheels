#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <memory>

#include "Dataset.h"

//basic constructor just takes in raw data and stores it in instance
Dataset::Dataset(std::vector<std::vector<float>> feature_data, std::vector<float> label_data) {
    //first check if number of feature and label entries are same, throw error if not
    if(feature_data.size() != label_data.size()){
        throw std::invalid_argument("feature and label sets must be same size.");
    }

    //must be initialized with some data.
    if(feature_data.size() == 0){
        throw std::invalid_argument("Dataset must be provided some data to be initialized.");   
    }

    //check if all feature data entries are of same dimension, throw error if not.
    for(int i = 1; i < feature_data.size(); i++){
        if (feature_data[i].size() != feature_data[0].size()) {
            throw std::invalid_argument("feature data must be of uniform dimension.");
        }
    }
    
    this->feature_data = std::make_unique<std::vector<std::vector<float>>>(feature_data);
    this->label_data = std::make_unique<std::vector<float>>(label_data);
    this->dataset_size = this->feature_data->size();
    this->num_features = this->feature_data->at(0).size();
}

//getters and setters
std::vector<std::vector<float>> Dataset::get_feature_data(){
    return *(this->feature_data);
}

std::vector<float> Dataset::get_label_data(){
    return *(this->label_data);
}

int Dataset::get_dataset_size() {
    return this->dataset_size;
}

int Dataset::get_num_features() {
    return this->num_features;
}

//adds feature-label pair to dataset
void Dataset::add_data_pair(std::vector<float> features, float label){
    //check if feature is of appropriate dimension, throw error if not
    if(features.size() != this->num_features){
        throw std::invalid_argument("Dimension of passed feature doesn't match that of dataset.");
    }

    this->feature_data->push_back(features);
    this->label_data->push_back(label);
}

//adds multiple feature-label pairs to dataset
void Dataset::add_data_pairs(std::vector<std::vector<float>> features, std::vector<float> labels){
    //check features are all of right size
    for(int i = 0; i < features.size(); i++){
        if(features[i].size() != this->num_features){
            throw std::invalid_argument("All passed in feature sets must be matching dimension of dataset.");
        }
    }

    //check that number of features and labels matches 
    if(features.size() != labels.size()){
        throw std::invalid_argument("number of feature sets to add to dataset must be same as number of labels.");
    }

    //add data
    for(int i = 0; i < features.size(); i++){
        this->add_data_pair(features[i], labels[i]);
    }
}

//removes feature-label pair given index
void Dataset::remove_data_pair(int index){
    //check if index is in bounds, throw error if not
    if(index < 0 || index >= this->dataset_size){
        throw std::invalid_argument("Index out of range.");
    }

    this->feature_data->erase(this->feature_data->begin() + index);
    this->label_data->erase(this->label_data->erase(this->label_data->begin() + index));
    this->dataset_size--;
}

//returns zipped feature and label data
std::vector<std::vector<std::vector<float>>> Dataset::zip_data(){
    std::vector<std::vector<std::vector<float>>> zipped_data = {};
    for(int i = 0; i < this->dataset_size; i++){
        zipped_data.push_back({this->feature_data->at(i), {this->label_data->at(i)}});
    }
    return zipped_data;
}

//shuffles dataset
//todo: make this better
void Dataset::shuffle_dataset(){
    //first zip up features and labels
    std::vector<std::vector<std::vector<float>>> zipped_data = this->zip_data();

    //initalize random generator we'll be using
    std::random_device rd;
    std::mt19937 gen(rd());
    //shuffle zipped array
    std::shuffle(zipped_data.begin(), zipped_data.end(), gen);
    //put back into base arrays
    this->feature_data->clear();
    this->label_data->clear();
    for(int i = 0; i < this->dataset_size; i++){
        this->add_data_pair(zipped_data[i][0], zipped_data[i][1][0]);
    }
}

