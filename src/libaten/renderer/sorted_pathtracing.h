#pragma once

#include "renderer/pathtracing.h"
#include "math/vec4.h"

namespace aten
{
	class SortedPathTracing : public PathTracing {
	public:
		SortedPathTracing() {}
		~SortedPathTracing() {}

		virtual void render(
			Destination& dst,
			scene* scene,
			camera* camera) override final;

	private:
		struct Path : public PathTracing::Path {
			CameraSampleResult camsample;
			real camSensitivity;

			uint32_t x, y;

			vec3 orienting_normal;

			sampler* sampler{ nullptr };

			real pdfLight{ real(0) };
			real dist2ToLight{ real(0) };
			real cosLight{ real(0) };
			real lightSelectPdf{ real(0) };
			LightAttribute lightAttrib;
			vec3 lightColor;
			
			struct {
				uint32_t isHit		: 1;
				uint32_t isAlive	: 1;
				uint32_t needWrite	: 1;
			};

			Path()
			{
				isHit = false;
				isAlive = true;
				needWrite = true;
			}
		};

		void makePaths(
			int width, int height,
			int sample,
			Path* paths,
			ray* rays,
			camera* camera);

		void hitPaths(
			Path* paths,
			const ray* rays,
			int numPath,
			scene* scene);

		void hitRays(
			ray* rays,
			int numRay,
			scene* scene);

		int compactionPaths(
			Path* paths,
			int numPath,
			uint32_t* hitIds);

		void shadeMiss(
			scene* scene,
			int depth,
			Path* paths,
			int numPath,
			vec4* dst);

		void shade(
			uint32_t sample,
			uint32_t depth,
			Path* paths,
			ray* rays,
			ray* shadowRays,
			uint32_t* hitIds,
			int numHit,
			camera* cam,
			scene* scene);

		void evalExplicitLight(
			Path* paths,
			const ray* shadowRays,
			uint32_t* hitIds,
			int numHit);

		void gather(
			Path* paths,
			int numPath,
			vec4* dst);

	private:
		uint32_t m_maxDepth{ 1 };

		// Depth to compute russinan roulette.
		uint32_t m_rrDepth{ 1 };

		uint32_t m_samples{ 1 };

		std::vector<vec4> m_tmpbuffer;
		int m_width{ 0 };
		int m_height{ 0 };
	};
}
