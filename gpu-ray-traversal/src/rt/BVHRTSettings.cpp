#include "BVHRTSettings.h"

#include <iostream>
#include <fstream>
#include <sstream>

BVHRTSettings::BVHRTSettings() : lbvh32(false), lbvh64(false), trbvh(false), atrbvh(false), 
        collapseTree(false), treeletSize(7), iterations(3)
{
    ParseConfigurationFile("bvhrt.cfg");
}

BVHRTSettings::~BVHRTSettings()
{
}

bool BVHRTSettings::Contains(std::vector<std::string> container, std::string token) const
{
    for (auto& element : container)
    {
        if (element == token)
        {
            return true;
        }
    }
    return false;
}

template<class T> bool BVHRTSettings::GetValue(std::vector<std::string> tokens, std::string id, 
        T& value) const
{
    bool found = false;

    for (auto& token : tokens)
    {
        if (token.find(id) != std::string::npos)
        {
            std::istringstream stream(token.substr(token.find('=') + 1));
            stream >> value;
            found = true;
        }
    }

    return found;
}

void BVHRTSettings::ParseConfigurationFile(char* fileLocation)
{
    std::ifstream file(fileLocation);
    std::string token;
    std::vector<std::string> tokens;
    if (file.is_open())
    {
        while (getline(file, token, ' '))
        {
            tokens.push_back(token);
        }
        file.close();
    }
    
    if (Contains(tokens, "lbvh"))
    {
        lbvh32 = true;
    }
    if (Contains(tokens, "lbvh64"))
    {
        lbvh64 = true;
    }
    if (Contains(tokens, "trbvh"))
    {
        trbvh = true;
    }
    if (Contains(tokens, "atrbvh"))
    {
        atrbvh = true;
    }   
    if (Contains(tokens, "collapse"))
    {
        collapseTree = true;
    }
    
    GetValue(tokens, "iterations", iterations);
    GetValue(tokens, "treeletSize", treeletSize);
}
