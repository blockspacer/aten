#pragma once

#include <vector>
#include "defs.h"
#include "types.h"
#include "math/vec3.h"
#include "math/vec4.h"

namespace aten {
	class shader;

	class texture {
		static std::vector<texture*> g_textures;

	public:
		texture();
		texture(uint32_t width, uint32_t height, uint32_t channels, const char* tag = nullptr);

		~texture();

	public:
		void init(uint32_t width, uint32_t height, uint32_t channels);

		AT_DEVICE_API vec3 at(real u, real v) const
		{
			u -= floor(u);
			v -= floor(v);

			uint32_t x = (uint32_t)(aten::cmpMin(u, real(1)) * (m_width - 1));
			uint32_t y = (uint32_t)(aten::cmpMin(v, real(1)) * (m_height - 1));

			uint32_t pos = y * m_width + x;

			const auto clr = m_colors[pos];

			vec3 ret;

			switch (m_channels) {
			case 3:
				ret[2] = clr[2];
			case 2:
				ret[1] = clr[1];
			case 1:
				ret[0] = clr[0];
				break;
			}

			return ret;
		}

		real& operator()(uint32_t x, uint32_t y, uint32_t c)
		{
			x = std::min(x, m_width - 1);
			y = std::min(y, m_height - 1);
			c = std::min(c, m_channels - 1);

			uint32_t pos = ((m_height - 1) - y) * m_width + x;

			return m_colors[pos][c];
		}

		const vec4* colors() const
		{
			return &m_colors[0];
		}

		uint32_t width() const
		{
			return m_width;
		}

		uint32_t height() const
		{
			return m_height;
		}

		uint32_t channels() const
		{
			return m_channels;
		}

		int id() const
		{
			return m_id;
		}

		static const texture* getTexture(int id);
		static const std::vector<texture*>& getTextures();

		bool initAsGLTexture();
		void bindAsGLTexture(uint8_t stage, shader* shd) const;
		void releaseAsGLTexture();
		void clearAsGLTexture(const aten::vec4& clearColor);

		void getDataAsGLTexture(
			int& width,
			int& height,
			int& channel,
			std::vector<vec4>& dst) const;

		static void bindAsGLTexture(
			uint32_t gltex,
			uint8_t stage, shader* shd);

		uint32_t getGLTexHandle() const
		{
			return m_gltex;
		}

	private:
		int m_id{ -1 };

		uint32_t m_width{ 0 };
		uint32_t m_height{ 0 };
		uint32_t m_channels{ 0 };

		uint32_t m_size{ 0 };

		std::vector<vec4> m_colors;

		uint32_t m_gltex{ 0 };

		std::string m_tag;
	};
}