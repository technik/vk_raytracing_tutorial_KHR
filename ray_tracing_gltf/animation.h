#pragma once

#include <nvmath/nvmath.h>

#include <chrono>
#include <vector>

// General multi channel animation
class Animation
{
public:
	template<class T>
	struct Track
	{
		std::vector<float> key;
		std::vector<T> value;
	};

	template<class T>
	class Sampler
	{
	public:
		Sampler(Track<T>& track)
			: m_track(track)
		{}

		void reset()
		{
			m_nextKey = 0;
			m_src = m_track.value[0];
			m_dst = m_track.value[0];
			m_t0 = 0.f;
			m_t1 = 0.f;
			m_t = 0.f;
		}
		void advance(std::chrono::duration<float> dt, T& out)
		{
			m_t += dt.count();
			while(m_t > m_t1)
			{
				// Advance slot
				m_nextKey = (m_nextKey + 1) % m_track.key.size(); // Loop around the animation
				if (m_nextKey == 0) // Just completed a loop
				{
					m_t -= m_t1;
					m_t1 = 0;
				}
				m_t0 = m_t1;
				m_src = m_dst;
				m_t1 = m_track.key[m_nextKey];
				m_dst = m_track.value[m_nextKey];
			}
			// interpolate
			float f = m_t == m_t0 ? 0 : (m_t - m_t0) / (m_t1 - m_t0);
			out = interpolate(f, m_src, m_dst);
		}

		inline static nvmath::vec3f interpolate(float f, const nvmath::vec3f& a, const nvmath::vec3f& b)
		{
			return lerp(f, a, b);
		}

		inline static float interpolate(float f, float a, float b)
		{
			return lerp(f, a, b);
		}

		inline static nvmath::quatf interpolate(float f, const nvmath::quatf& a, const nvmath::quatf& b)
		{
			return slerp_quats(f, a, b);
		}

	private:
		Track<T>& m_track;
		T m_src;
		T m_dst;
		size_t m_nextKey = 0;
		float m_t = 0;
		float m_t0 = 0;
		float m_t1 = 0;
	};

	using PositionTrack = Track<nvmath::vec3f>;
	using RotationTrack = Track<nvmath::quatf>;
	using ScaleTrack = Track<nvmath::vec3f>;

	Animation(
		const PositionTrack& positionTrack,
		const RotationTrack& rotationTrack,
		const ScaleTrack& scaleTrack,
		nvmath::mat4f& target)
		: m_mtx(target)
		, m_position(positionTrack)
		, m_rotation(rotationTrack)
		, m_scale(scaleTrack)
		, m_positionSampler(m_position)
		, m_rotationSampler(m_rotation)
		, m_scaleSampler(m_scale)
	{}

	// Timeline
	void reset() // Go back to t = 0
	{
		m_positionSampler.reset();
		m_rotationSampler.reset();
		m_scaleSampler.reset();
	}

	void advance(std::chrono::duration<float> dt)
	{
		nvmath::vec3f pos;
		nvmath::quatf rot;
		nvmath::vec3f scale;

		m_positionSampler.advance(dt, pos);
		m_rotationSampler.advance(dt, rot);
		m_scaleSampler.advance(dt, scale);

		nvmath::mat4f rotMtx = nvmath::quat_2_mat(rot);
		m_mtx = nvmath::translation_mat4(pos) * nvmath::scale_mat4(scale) * rotMtx;
	}

private:
	nvmath::mat4f& m_mtx;
	
	PositionTrack m_position;
	RotationTrack m_rotation;
	ScaleTrack m_scale;

	Sampler<nvmath::vec3f> m_positionSampler;
	Sampler<nvmath::quatf> m_rotationSampler;
	Sampler<nvmath::vec3f> m_scaleSampler;
};
