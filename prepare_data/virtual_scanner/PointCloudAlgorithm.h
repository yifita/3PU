#pragma once

#include "DataMgr.h"
#include "ParameterMgr.h"

class PointCloudAlgorithm
{
public:
	PointCloudAlgorithm(RichParameterSet* _para){}
	virtual ~PointCloudAlgorithm(){}

	virtual void setInput(DataMgr* pData) = 0;
	virtual void setParameterSet(RichParameterSet* _para) = 0;
	virtual RichParameterSet* getParameterSet() = 0;
	virtual void run() = 0;
	virtual void clear() = 0;

protected:
	PointCloudAlgorithm(){}

private:
	//static ParameterMgr para;
};