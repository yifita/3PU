#include<vcg/complex/complex.h>

#include<wrap/io_trimesh/import.h>
#include<wrap/io_trimesh/export.h>
#include<wrap/ply/plylib.h>

#include<vcg/complex/algorithms/point_sampling.h>
#include<vcg/complex/algorithms/create/platonic.h>

using namespace vcg;
using namespace std;

class MyEdge;
class MyFace;
class MyVertex;
struct MyUsedTypes : public UsedTypes<	Use<MyVertex>   ::AsVertexType,
	Use<MyEdge>     ::AsEdgeType,
	Use<MyFace>     ::AsFaceType> {};

class MyVertex : public Vertex<MyUsedTypes, vertex::Coord3f, vertex::Normal3f, vertex::BitFlags  > {};
class MyFace : public Face< MyUsedTypes, face::FFAdj, face::Normal3f, face::VertexRef, face::BitFlags > {};
class MyEdge : public Edge<MyUsedTypes> {};
class MyMesh : public tri::TriMesh< vector<MyVertex>, vector<MyFace>, vector<MyEdge>  > {};

vector<string> split(const string &s, const string &seperator) {
	vector<string> result;
	typedef string::size_type string_size;
	string_size i = 0;

	while (i != s.size()) {

		int flag = 0;
		while (i != s.size() && flag == 0) {
			flag = 1;
			for (string_size x = 0; x < seperator.size(); ++x)
				if (s[i] == seperator[x]) {
					++i;
					flag = 0;
					break;
				}
		}


		flag = 0;
		string_size j = i;
		while (j != s.size() && flag == 0) {
			for (string_size x = 0; x < seperator.size(); ++x)
				if (s[j] == seperator[x]) {
					flag = 1;
					break;
				}
			if (flag == 0)
				++j;
		}
		if (i != j) {
			result.push_back(s.substr(i, j - i));
			i = j;
		}
	}
	return result;
}

int main(int argc, char **argv)
{
	MyMesh m;
	typedef tri::MeshSampler<MyMesh> BaseSampler;

	if (tri::io::Importer<MyMesh>::Open(m, argv[2]))
	{
		printf("Error reading file  %s\n", argv[2]);
		exit(0);
	}
	tri::SurfaceSampling<MyMesh, tri::TrivialSampler<MyMesh> >::SamplingRandomGenerator().initialize(time(0));
	vector<Point3f> pointVec;
	int sampleNum = atoi(argv[1]);
	float radius = tri::SurfaceSampling<MyMesh, BaseSampler>::ComputePoissonDiskRadius(m, sampleNum);
	tri::PoissonSampling<MyMesh>(m, pointVec, sampleNum, radius);

	FILE *fp;
	char* gtModelName = argv[3];
	// tri::io::ExporterPLY<MyMesh>::Save(pointVec, gtModelName, true);
	fp = fopen(gtModelName, "w");
	for (int ptNum = 0; ptNum < pointVec.size(); ptNum++)
	{
		fprintf(fp, "%f %f %f\n", pointVec[ptNum].X(), pointVec[ptNum].Y(), pointVec[ptNum].Z());
	}
	fclose(fp);
	cout << atoi(argv[1]) << " "<<pointVec.size() << endl;

	return 0;
}
