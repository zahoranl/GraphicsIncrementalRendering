#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif

const unsigned int windowWidth = 600, windowHeight = 600;
//const unsigned int windowWidth = 1080, windowHeight = 1080;

int majorVersion = 3, minorVersion = 3;
//---------------------------
struct vec2 {
	//---------------------------
	float x, y;
	vec2(float _x = 0, float _y = 0) { x = _x; y = _y; }
};
//---------------------------
struct vec3 {
	//---------------------------
	float x, y, z;
	vec3(float _x = 0, float _y = 0, float _z = 0) { x = _x; y = _y; z = _z; }
	vec3 operator*(float a) const { return vec3(x * a, y * a, z * a); }
	vec3 operator+(const vec3& v) const { return vec3(x + v.x, y + v.y, z + v.z); }
	vec3 operator-(const vec3& v) const { return vec3(x - v.x, y - v.y, z - v.z); }
	vec3 operator*(const vec3& v) const { return vec3(x * v.x, y * v.y, z * v.z); }
	vec3 operator/(const vec3& v) const { return vec3(x / v.x, y / v.y, z / v.z); }
	vec3 operator-() const { return vec3(-x, -y, -z); }
	vec3 normalize() const { return (*this) * (1.0f / (Length() + 0.000001)); }
	float Length() const { return sqrtf(x * x + y * y + z * z); }

	void SetUniform(unsigned shaderProg, char * name) {
		int location = glGetUniformLocation(shaderProg, name);
		if (location >= 0) glUniform3fv(location, 1, &x);
		else printf("uniform %s cannot be set\n", name);
	}
};
float dot(const vec3& v1, const vec3& v2) { return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z); }
vec3 cross(const vec3& v1, const vec3& v2) { return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x); }
//---------------------------
struct mat4 { // row-major matrix 4x4
			  //---------------------------
	float m[4][4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	mat4 operator*(const mat4& right) {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}
	void SetUniform(unsigned shaderProg, char * name) {
		int location = glGetUniformLocation(shaderProg, name);
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, &m[0][0]);
		else printf("uniform %s cannot be set\n", name);
	}
};
mat4 TranslateMatrix(vec3 t) {
	return mat4(1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		t.x, t.y, t.z, 1);
}
mat4 ScaleMatrix(vec3 s) {
	return mat4(s.x, 0, 0, 0,
		0, s.y, 0, 0,
		0, 0, s.z, 0,
		0, 0, 0, 1);
}
mat4 RotationMatrix(float angle, vec3 w) {
	float c = cosf(angle), s = sinf(angle);
	w = w.normalize();
	return mat4(c * (1 - w.x*w.x) + w.x*w.x, w.x*w.y*(1 - c) + w.z*s, w.x*w.z*(1 - c) - w.y*s, 0,
		w.x*w.y*(1 - c) - w.z*s, c * (1 - w.y*w.y) + w.y*w.y, w.y*w.z*(1 - c) + w.x*s, 0,
		w.x*w.z*(1 - c) + w.y*s, w.y*w.z*(1 - c) - w.x*s, c * (1 - w.z*w.z) + w.z*w.z, 0,
		0, 0, 0, 1);
}
//---------------------------
struct Camera { 
	vec3 wEye, wLookat, wVup;
	float fov, asp, fp, bp;
public:
	Camera() {
		asp = 1;
		fov = 60.0f * (float)M_PI / 180.0f;
		fp = 1; bp = 100;
	}
	mat4 V() { // view matrix: translates the center to the origin
		vec3 w = (wEye - wLookat).normalize();
		vec3 u = cross(wVup, w).normalize();
		vec3 v = cross(w, u);
		return TranslateMatrix(-wEye) * mat4(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}
	mat4 P() {
		return mat4( 1 , 0, 0, 0,
					0, 1, 0, 0,
					0, 0, 0, 0,
				0, 0, 0, 1 );

		return mat4(1 / (tan(fov / 2)*asp), 0, 0, 0,
					0, 1 / tan(fov / 2), 0, 0,
					0, 0, -(fp + bp) / (bp - fp), -1,
					0, 0, -2 * fp*bp / (bp - fp), 0);
	}
	void Animate(float t) {
	}
};
//---------------------------
struct Material {
	//---------------------------
	vec3 kd, ks, ka;
	float shininess;
};
//---------------------------
struct Light {
	//---------------------------
	vec3 La, Le;
	vec3 wLightPos;

	Light() {
		La = vec3(1, 1, 1);
		Le = vec3(3,3, 3);
	}
	void Animate(float t, float dt) {	}
};
//---------------------------
struct Texture {
	//---------------------------
	unsigned int textureId, textureUnit;

	Texture(const int width = 0, const int height = 0,vec3 c1=vec3(0,0,0), vec3 c2=vec3(1,1,1),bool random=false) {
		textureUnit = 2;	// it can be 0, 1, ... depending on how many texture filtering units we have
		glGenTextures(1, &textureId);  				// id generation
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding
		std::vector<vec3> image(width * height);

		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x & 1) ^ (y & 1) ? c1 : c2;
			if (random) {
				srand(glutGet(GLUT_ELAPSED_TIME)*x*y);
				if (rand() % 3 == 0)  image[y * width + x] = vec3(0,0.36,0.035);
			}
		}
		


		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, &image[0]); //Texture->OpenGL
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}

	void SetUniform(unsigned shaderProg, char * name) {
		int location = glGetUniformLocation(shaderProg, name);
		if (location >= 0) {
			glUniform1i(location, textureUnit);
			glActiveTexture(GL_TEXTURE0 + textureUnit);
			glBindTexture(GL_TEXTURE_2D, textureId);
		}
		//else printf("uniform %s cannot be set\n", name);
	}
};
//---------------------------
struct RenderState {
	//---------------------------
	mat4	  MVP, M, Minv;
	Material* material;
	Light     light1;
	Light     light2;
	Light     WayLight;
	Texture*  texture;
	vec3	  wEye;
} state;
//---------------------------
class Shader {
	//--------------------------
	void getErrorInfo(unsigned int handle) {
		int logLen, written;
		glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
		if (logLen > 0) {
			char * log = new char[logLen];
			glGetShaderInfoLog(handle, logLen, &written, log);
			printf("Shader log:\n%s", log);
			delete log;
		}
	}
	void checkShader(unsigned int shader, char * message) { 	// check if shader could be compiled
		int OK;
		glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
		if (!OK) { printf("%s!\n", message); getErrorInfo(shader); getchar(); }
	}
	void checkLinking(unsigned int program) { 	// check if shader could be linked
		int OK;
		glGetProgramiv(program, GL_LINK_STATUS, &OK);
		if (!OK) { printf("Failed to link shader program!\n"); getErrorInfo(program); getchar(); }
	}
protected:
	unsigned int shaderProgram;
public:
	void Create(const char * vertexSource, const char * fragmentSource, const char * fsOuputName) {
		// Create vertex shader from string
		unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
		if (!vertexShader) { printf("Error in vertex shader creation\n"); exit(1); }
		glShaderSource(vertexShader, 1, &vertexSource, NULL);
		glCompileShader(vertexShader);
		checkShader(vertexShader, "Vertex shader error");

		// Create fragment shader from string
		unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		if (!fragmentShader) { printf("Error in fragment shader creation\n"); exit(1); }
		glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
		glCompileShader(fragmentShader);
		checkShader(fragmentShader, "Fragment shader error");

		// Attach shaders to a single program
		shaderProgram = glCreateProgram();
		if (!shaderProgram) { printf("Error in shader program creation\n"); exit(1); }
		glAttachShader(shaderProgram, vertexShader);
		glAttachShader(shaderProgram, fragmentShader);

		// Connect the fragmentColor to the frame buffer memory
		glBindFragDataLocation(shaderProgram, 0, fsOuputName);	// fragmentColor goes to the frame buffer memory

																// program packaging
		glLinkProgram(shaderProgram);
		checkLinking(shaderProgram);
	}
	virtual void Bind() = 0;
	~Shader() { glDeleteProgram(shaderProgram); }
};
//---------------------------
class PhongShader : public Shader {
	//---------------------------
	const char * vertexSource = R"(
		#version 330
		precision highp float;

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform vec3  wLiPos1;       // light source direction 
		uniform vec3  wLiPos2;       // light source direction 
		uniform vec3  wLiPosW;       // light source direction 
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight1;		    // light dir in world space
		out vec3 wLight2;		    // light dir in world space
		out vec3 wLightW;		    // light dir in world space

		out vec2 texcoord;

		void main() {
		   gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
		   vec4 wPos = vec4(vtxPos, 1) * M;
		   wLight1  = wLiPos1 - wPos.xyz;
		   wLight2  = wLiPos2 - wPos.xyz;
		   wLightW  = wLiPosW - wPos.xyz;
		   wView   = wEye - wPos.xyz;
		   wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		   texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char * fragmentSource = R"(
		#version 330
		precision highp float;

		uniform vec3 kd, ks, ka; // diffuse, specular, ambient ref
		uniform vec3 La1, Le1;     // ambient and point sources
		uniform vec3 La2, Le2;     
		uniform vec3 LaW, LeW;     
		uniform float shine;     // shininess for specular ref
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight1;        // interpolated world sp illum dir
		in  vec3 wLight2;        // interpolated world sp illum dir
		in  vec3 wLightW;        // interpolated world sp illum dir

		in vec2 texcoord;
		out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			vec3 L1 = normalize(wLight1);
			vec3 L2 = normalize(wLight2);
			vec3 LW = normalize(wLightW);

			vec3 H1 = normalize(L1 + V);
			vec3 H2 = normalize(L2 + V);
			vec3 HW = normalize(LW + V);

			float cost1 = max(dot(N,L1), 0), cosd1 = max(dot(N,H1), 0);
			float cost2 = max(dot(N,L2), 0), cosd2 = max(dot(N,H2), 0);
			float costW = max(dot(N,LW), 0), cosdW = max(dot(N,HW), 0);

			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 color = ka * texColor * La1 + (kd * texColor * cost1 + ks * pow(cosd1,shine)) * Le1;
			color=color+ ka * texColor * La2 + (kd * texColor * cost2 + ks * pow(cosd2,shine)) * Le2;
			color=color+ ka * texColor * LaW + (kd * texColor * costW + ks * pow(cosdW,shine)) * LeW;
			fragmentColor = vec4(color, 1);
		}
	)";
public:
	PhongShader() { Create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind() {
		glUseProgram(shaderProgram); 		// make this program run
		state.MVP.SetUniform(shaderProgram, "MVP");
		state.M.SetUniform(shaderProgram, "M");
		state.Minv.SetUniform(shaderProgram, "Minv");
		state.wEye.SetUniform(shaderProgram, "wEye");
		state.material->kd.SetUniform(shaderProgram, "kd");
		state.material->ks.SetUniform(shaderProgram, "ks");
		state.material->ka.SetUniform(shaderProgram, "ka");
		int location = glGetUniformLocation(shaderProgram, "shine");
		if (location >= 0) glUniform1f(location, state.material->shininess); else printf("uniform shininess cannot be set\n");
		state.light1.La.SetUniform(shaderProgram, "La1");
		state.light1.Le.SetUniform(shaderProgram, "Le1");
		state.light1.wLightPos.SetUniform(shaderProgram, "wLiPos1");
		
		state.light2.La.SetUniform(shaderProgram, "La2");
		state.light2.Le.SetUniform(shaderProgram, "Le2");
		state.light2.wLightPos.SetUniform(shaderProgram, "wLiPos2");
		
		state.WayLight.La.SetUniform(shaderProgram, "LaW");
		state.WayLight.Le.SetUniform(shaderProgram, "LeW");
		state.WayLight.wLightPos.SetUniform(shaderProgram, "wLiPosW");

		state.texture->SetUniform(shaderProgram, "diffuseTexture");
	}
};
//---------------------------
struct VertexData {
	//---------------------------
	vec3 position, normal;
	vec2 texcoord;
};
//---------------------------
class Geometry {
	//---------------------------
	unsigned int vao, type;        // vertex array object
protected:
	int nVertices;
public:
	Geometry(unsigned int _type) {
		type = _type;
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
	}
	void Draw() {
		glBindVertexArray(vao);
		glDrawArrays(type, 0, nVertices);
	}
};
//---------------------------
class ParamSurface : public Geometry {
	//---------------------------
public:
	ParamSurface() : Geometry(GL_TRIANGLES) {}

	virtual VertexData GenVertexData(float u, float v) = 0;

	void Create(int N = 16, int M = 16, std::vector<VertexData>* vtxDataf = NULL, int d=0) {
		unsigned int vbo;
		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		std::vector<VertexData> vtxData;// vertices on the CPU
		if (vtxDataf==NULL){
			nVertices = N* M * 6;
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < M; j++) {
					vtxData.push_back(GenVertexData((float)i / N, (float)j / M));
					vtxData.push_back(GenVertexData((float)(i + 1) / N, (float)j / M));
					vtxData.push_back(GenVertexData((float)i / N, (float)(j + 1) / M));
					vtxData.push_back(GenVertexData((float)(i + 1) / N, (float)j / M));
					vtxData.push_back(GenVertexData((float)(i + 1) / N, (float)(j + 1) / M));
					vtxData.push_back(GenVertexData((float)i / N, (float)(j + 1) / M));
				}
			}
		}else
		{
			vtxData = *vtxDataf;
			nVertices = d;
		}

		glBufferData(GL_ARRAY_BUFFER, nVertices * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
									   // attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}
};
class LagrangeCurve {
public:
	std::vector<vec3>  cps; // control pts 
	std::vector<float> ts;  // knots

	float L(int i, float t) {
		float Li = 1.0f;
		for (int j = 0; j < cps.size(); j++)
			if (j != i) Li = Li * (t - ts[j]) / (ts[i] - ts[j]);
		return Li;
	}
	vec3 r(float t) {
		vec3 rr(0, 0, 0);
		for (int i = 0; i<cps.size(); i++) rr = rr + (cps[i] * L(i, t));
		return rr;
	}
};
LagrangeCurve curve;
LagrangeCurve curve2;
//---------------------------
class Triangle : public ParamSurface {
	//---------------------------
public:
	
	std::vector<VertexData> vtxData;
	Triangle() {
		CreateSpine();
		
		//for (float i = 0.0; i <= 2.2; i = i + 0.1) printf("x: %f \t y:°%f \t z:%f \t i:%f \n", curve.r(i).x, curve.r(i).y, curve.r(i).z, i);
	}


	void CreateSpine() {
		float up = 0.275;//0.55
		float right = 0.2;//0.4
		curve.ts.push_back(0);
		curve.cps.push_back(vec3(1.0, 0.0, 0.0));
		curve.ts.push_back(1);
		curve.cps.push_back(vec3(0.2, 0.3, 0.0));
		curve.ts.push_back(2);
		curve.cps.push_back(vec3(0.0, 1.0, 0.0));

		curve2.ts.push_back(0);
		curve2.cps.push_back(vec3(0.0 + right, 1.0 + up, 0.0));
		curve2.ts.push_back(1);
		curve2.cps.push_back(vec3(0.2 + right, 0.3 + up, 0.0));
		curve2.ts.push_back(2);
		curve2.cps.push_back(vec3(1.0 + right, 0.0 + up, 0.0));
		ReCreate();
	}

	void EditSpine(float x,float y) {
		curve.cps[1].x = x;
		curve.cps[1].y = y;
		curve2.cps[1].x = x+0.2;
		curve2.cps[1].y = y+0.275;
		ReCreate();
	}

	void ReCreate() {

		std::vector<VertexData> vtxData;
		for (float u = 0.0; u <= 1.0; u = u + 0.05) {
			for (float v = 0.0; v <= 1.0; v = v + 1.0) {
				vtxData.push_back(GenVertexData(u, v));
				vtxData.push_back(GenVertexData(u + 0.1, v));
				vtxData.push_back(GenVertexData(u, v + 1.0));
				vtxData.push_back(GenVertexData(u + 0.1, v));
				vtxData.push_back(GenVertexData(u + 0.1, v + 1.0));
				vtxData.push_back(GenVertexData(u, v + 1.0));
			}
		}
		Create(0, 0, &vtxData, 20 * 2 * 6);
	}

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		if (v == 0.0) vd.position = vec3(v, u, 0);
		else vd.position = vec3(curve.r(u*2.2).x, curve.r(u*2.2).y, 0);
		vd.normal = vec3(0, 0, 1.0);
		vd.texcoord = vec2(u, v);
		return vd;
	}

};
//---------------------------
class Torus : public ParamSurface {
	//---------------------------
	const float R = 1, r = 0.5;

	vec3 Point(float u, float v, float rr) {
		float ur = u * 2.0f * M_PI, vr = v * 2.0f * M_PI;
		float l = R + rr * cos(ur);
		return vec3(l * cos(vr), l * sin(vr), rr * sin(ur));
	}
public:
	Torus() { Create(40, 40); }

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vd.position = Point(u, v, r);
		vd.normal = (vd.position - Point(u, v, 0)) * (1.0f / r);
		vd.texcoord = vec2(u, v);
		return vd;
	}
};
//---------------------------
struct Object {
	//---------------------------
	Shader * shader;
	Material * material;
	Texture * texture;
	Geometry * geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
	bool animate = false;
public:
	Object(Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = _geometry;
	}
	void Draw(Camera& camera) {
		state.M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		state.Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
		state.MVP = state.M * camera.V() * camera.P();
		state.material = material;
		state.texture = texture;
		shader->Bind();
		geometry->Draw();
	}

	float u = -1, v = 1;
	float time = 0;

	void Animate(float t, float dt) { 

		rotationAngle =  -4*time; 
		translation=curve2.r(time)+ vec3(-1.0, -1.0+0.3, 0.0);
		translation = curve2.r(time) + vec3(-0.8+sqrtf((0.3*0.3)/2), -0.7 + sqrtf((0.3*0.3) / 2), 0.0);

		translation = curve2.r(time)+vec3(-1.0,-1.0,0.0);

		time = time + 0.001;
		if (time > 2.2) {
			printf("%f sec \n", time);
			time = 0; 
		}

	/*	u=u + 0.1*(dt / 100);
		v=v - 0.1*(dt / 100);
		time++;
		if (v < -1.2) {
			u = -1;
			v = 1; 
			printf("time:= %i", time);
			time = 0;
		}*/
		
	}
};
//---------------------------
Geometry * triangle;
class Scene {
	//---------------------------
	std::vector<Object *> objects;
public:
	Camera camera; // 3D camera
	Light lightPoint1;
	Light lightPoint2;
	Light lightWay;

	void Build() {
		// Shaders
		Shader * phongShader = new PhongShader();

		// Materials
		Material * material0 = new Material;
		material0->kd = vec3(1.0f, 0.1f, 0.2f);
		material0->ks = vec3(1, 1, 1);
		material0->ka = vec3(0.2f, 0.2f, 0.2f);
		material0->shininess = 50;

		Material * material1 = new Material;
		material1->kd = vec3(0, 1, 1);
		material1->ks = vec3(1, 1, 1);
		material1->ka = vec3(0.2f, 0.2f, 0.2f);
		material1->shininess = 200;

		// Textures
		Texture * texture4x8 = new Texture(4, 8,vec3(1, 1, 0), vec3(1, 0, 0.2));
		Texture * texture15x20 = new Texture(1000,1000 ,vec3(0, 0.85, 0.06), vec3(0, 0.48, 0.047),true);

		// Geometries
		Geometry * torus = new Torus();
		triangle = new Triangle();
	

		// Create objects by setting up their vertex data on the GPU
		Object * torusObject1 = new Object(phongShader, material0, texture4x8, torus);
		torusObject1->translation = vec3(0, 0, 0);
		torusObject1->rotationAxis = vec3(0, 0, 1);
		torusObject1->scale = vec3(0.2f, 0.2f, 0.2f);
		AddObject(torusObject1);


		Object * TriangleObject = new Object(phongShader, material1, texture15x20, triangle);
		//TriangleObject->translation = vec3(-2.5, -2.5, 2);
		TriangleObject->translation = vec3(-1.0,-1.0, 0.0);
		TriangleObject->rotationAxis = vec3(0, 0, 0);
		TriangleObject->scale = vec3(2, 2 , 2);

		//TriangleObject->scale = vec3(1, 1, 1);
		//TriangleObject->translation = vec3(0,0, 0);

		AddObject(TriangleObject);
		torusObject1->animate = true;
		//TriangleObject->animate = true;
		
		// Camera
		camera.wEye = vec3(0, 0,6);
		camera.wLookat = vec3(0, 0, 0);
		camera.wVup = vec3(0, 1, 0);

		// Light
		lightPoint1.wLightPos = vec3(5, 5, 4);
		lightPoint1.wLightPos = vec3(-5, -9, 4);
		lightWay.wLightPos = vec3(0, 100, 5);

	}
	void Render() {
		state.wEye = camera.wEye;
		state.light1 = lightPoint1;
		state.light2 = lightPoint2;
		state.WayLight = lightWay;
		for (Object * obj : objects) obj->Draw(camera);
	}

	void AddObject(Object * obj) { objects.push_back(obj); }
	void Edit(float x, float y) {
		//EditSpine(x, y);
	}
	void Animate(float t) {
		static float tprev = 0;
		float dt = t - tprev;
		camera.Animate(t);
		lightPoint1.Animate(t, dt);
		lightPoint2.Animate(t, dt);

		for (Object * obj : objects) {
			if (obj->animate){
			obj->Animate(t, dt);
			}
		}
	}
};
Scene scene;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}
// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.52f, 0.8f, 0.98f, 1.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	scene.Render();
	glutSwapBuffers();									// exchange the two buffers
}
// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) { }
void onExit() {
	printf("exit");
}
// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) { }
// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		float cX = 2.0f * pX / windowWidth - 1;
		float cY = 1.0f - 2.0f * pY / windowHeight;
		printf("x: %f,  y:%f \n", cX, cY);
		((Triangle*)triangle)->EditSpine(cX,cY);
		glutPostRedisplay();
	}
}
// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}
// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 10000.0f;	// convert msec to sec
	scene.Animate(sec);					// animate the camera
	glutPostRedisplay();					// redraw the scene
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Do not touch the code below this line

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_3_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}

