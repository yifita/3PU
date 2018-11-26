#pragma once
#include "Parameter.h"
#include "CMesh.h"

class ParameterMgr
{
public:
	ParameterMgr(void);
	~ParameterMgr(void);
  RichParameterSet* getDataParameterSet()              { return &data; }
  RichParameterSet* getDrawerParameterSet()            { return &drawer; }
  RichParameterSet* getGlareaParameterSet()            { return &glarea; }
  RichParameterSet* getNormalSmootherParameterSet()    { return &norSmooth; }
  RichParameterSet* getPoissonParameterSet()           { return &poisson; }
  RichParameterSet* getCameraParameterSet()            { return &camera; }
  RichParameterSet* getNBVParameterSet()               { return &nbv;   }

	void setGlobalParameter(QString paraName,Value& val);
	typedef enum {GLAREA, DATA, DRAWER, NOR_SMOOTH, POISSON}ParaType;

private:
	void initDataMgrParameter();
	void initDrawerParameter();
	void initGlareaParameter();
	void initNormalSmootherParameter();
  void initPoissonParameter();
  void initCameraParameter();
  void initNBVParameter();

public:
	RichParameterSet glarea;
	RichParameterSet data;
	RichParameterSet drawer;
	RichParameterSet norSmooth;
  RichParameterSet poisson;
  RichParameterSet camera;
  RichParameterSet nbv;

private:
	static int init_time;
	double grid_r;
};

extern ParameterMgr global_paraMgr;
