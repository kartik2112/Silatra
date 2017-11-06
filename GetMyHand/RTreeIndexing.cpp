/**
* This file holds the code for interacting 
* with the RTree data structures created in 
* libspatialindex library
* Code referenced from 
* https://github.com/libspatialindex/libspatialindex/wiki/4.-Nearest-neighbour-search
* Class names and member function names found 
* from documentation located at
* http://libspatialindex.github.io/doxygen/index.html
*/

#include <iostream>
#include <spatialindex/capi/sidx_api.h>
#include <spatialindex/capi/sidx_impl.h>
#include <spatialindex/capi/sidx_config.h>

using namespace std;
using namespace SpatialIndex;

// function to create a new spatial index
Index* createIndex()
{
	// create a property set with default values.
	// see utility.cc for all defaults  http://libspatialindex.github.io/doxygen/Utility_8cc_source.html#l00031
	Tools::PropertySet* ps = GetDefaults();
	Tools::Variant var;

	// set index type to R*-Tree
	var.m_varType = Tools::VT_ULONG;
	var.m_val.ulVal = RT_RTree;
	ps->setProperty("IndexType", var);

	// Set index to store in memory (default is disk)
	var.m_varType = Tools::VT_ULONG;
	var.m_val.ulVal = RT_Memory;
	ps->setProperty("IndexStorageType", var);

	// initalise index
	Index* idx = new Index(*ps);
	delete ps;

	// check index is ok
	if (!idx->index().isIndexValid())
		throw "Failed to create valid R-Tree index";
	else
		cout << "created R-Tree index" << endl;

	return idx;
}

// add a Point to index.
void addPoint(Index* idx,double lat,double lon, int64_t id)
{
	// create array with lat/lon points
	double coords[] = {lat, lon};

	// shapes can also have anobject associated with them but we'll leave that for the moment.
	uint8_t* pData = 0;
	uint32_t nDataLength = 0;

	// create shape
	SpatialIndex::IShape* shape = 0;
	shape = new SpatialIndex::Point(coords, 2);

	// insert into index along with the an object and an ID
	idx->index().insertData(nDataLength,pData,*shape,id);

	cout << "Point " << id << " inserted into index." << endl;

	delete shape;

}


// remove a Point to index.
void deletePoint(Index* idx,double lat,double lon, int64_t id)
{
	// create array with lat/lon points
	double coords[] = {lat, lon};

	// shapes can also have anobject associated with them but we'll leave that for the moment.
	uint8_t* pData = 0;
	uint32_t nDataLength = 0;

	// create shape
	SpatialIndex::IShape* shape = 0;
	shape = new SpatialIndex::Point(coords, 2);

	// insert into index along with the an object and an ID
	idx->index().deleteData(*shape,id);

	cout << "Point " << id << " deleted from index." << endl;

	delete shape;

}

/* 
This function is used to get the nearest points to the provided point
Here a square region around this point is considered for 
finding out nearest points
*/
std::vector<SpatialIndex::IData*>* getNearest(Index* idx,double lat,double lon)
{
    int sizeOfRectBy2 = 15;
    double coordsLow[] = {lat-sizeOfRectBy2,lon-sizeOfRectBy2};
    double coordsHigh[] = {lat+sizeOfRectBy2,lon+sizeOfRectBy2};
    cout<<"Searching for points in {"<<coordsLow[0]<<","<<coordsLow[1]<<"} to {"<<coordsHigh[0]<<","<<coordsHigh[1]<<"}"<<endl;
	// get a visitor object and a point from which to search
    ObjVisitor* visitor = new ObjVisitor;
    
	// get nearest maxResults shapes from index    
    SpatialIndex::Region* reg = new SpatialIndex::Region(coordsLow,coordsHigh,2);
    idx->index().intersectsWithQuery(*reg,*visitor);

	// get count of results
	int64_t nResultCount;
	nResultCount = visitor->GetResultCount();

	// get actual results
	std::vector<SpatialIndex::IData*>& results = visitor->GetResults();
	// an empty vector that we will copy the results to
	vector<SpatialIndex::IData*>* resultsCopy = new vector<SpatialIndex::IData*>();

	// copy the Items into the newly allocated vector array
	// we need to make sure to clone the actual Item instead
	// of just the pointers, as the visitor will nuke them
	// upon destroy
	for (int64_t i = 0; i < nResultCount; ++i)
	{
		resultsCopy->push_back(dynamic_cast<SpatialIndex::IData*>(results[i]->clone()));
	}

	delete reg;
	delete visitor;
	cout << "found " << nResultCount << " results." << endl;

	return resultsCopy;
}

int main(int argc, char* argv[])
{
	// initalise Index pointer
	Index* idx = createIndex();
    
    // add some points
    addPoint(idx,10,10,1); // buckingham palace
    addPoint(idx,20,20,2); // tower bridge
    addPoint(idx,15,15,3); // hyde park
    deletePoint(idx,15,15,3);
    // get nearest two locations to the royal albert hall
    std::vector<SpatialIndex::IData*>* results = getNearest(idx,0,0);
    
    // for each item
	for (SpatialIndex::IData* &item : (*results))
	{

		// get the generic shape object which we can cast to other types
		SpatialIndex::IShape* shape;
		item->getShape(&shape);

		// cast the shape to a Point
		SpatialIndex::Point center;
		shape->getCenter(center);

		//get ID of shape
		id_type id = item->getIdentifier();
        cout<<id<<endl;
	}
    
}