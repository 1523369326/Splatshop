#pragma once

#include <algorithm>
#include <cmath>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

// 项目内置 GLM 较旧，四元数的部分标量数学函数与新版 MSVC 不兼容。
// 相机模块在这里直接完成归一化、轴角转换和球面插值，避免修改全局 GLM。
inline float cameraQuaternionLength(const glm::quat& value) {
	return std::sqrt(value.w * value.w + value.x * value.x + value.y * value.y + value.z * value.z);
}

inline glm::quat cameraNormalizeQuaternion(const glm::quat& value) {
	float length = cameraQuaternionLength(value);
	if(!std::isfinite(length) || length <= 1e-8f) return glm::quat(1, 0, 0, 0);
	return glm::quat(value.w / length, value.x / length, value.y / length, value.z / length);
}

inline glm::quat cameraAngleAxis(float angle, const glm::vec3& axis) {
	float length = std::sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
	if(!std::isfinite(length) || length <= 1e-8f) return glm::quat(1, 0, 0, 0);
	float half = angle * 0.5f;
	float scale = std::sin(half) / length;
	return glm::quat(std::cos(half), axis.x * scale, axis.y * scale, axis.z * scale);
}

inline glm::quat cameraSlerp(glm::quat start, glm::quat end, float amount) {
	start = cameraNormalizeQuaternion(start);
	end = cameraNormalizeQuaternion(end);
	float cosine = start.w * end.w + start.x * end.x + start.y * end.y + start.z * end.z;
	if(cosine < 0) {
		end = -end;
		cosine = -cosine;
	}
	cosine = std::clamp(cosine, -1.0f, 1.0f);
	if(cosine > 0.9995f) return cameraNormalizeQuaternion(start * (1.0f - amount) + end * amount);
	float angle = std::acos(cosine);
	float divisor = std::sin(angle);
	if(std::abs(divisor) <= 1e-8f) return start;
	return cameraNormalizeQuaternion(
		start * (std::sin((1.0f - amount) * angle) / divisor) +
		end * (std::sin(amount * angle) / divisor));
}

struct CameraKeyframe {
	glm::vec3 position{0.0f};
	glm::quat orientation{1.0f, 0.0f, 0.0f, 0.0f};
	float fovy = 47.136211f;
	float secondsToNext = 2.0f;

	glm::mat4 world() const {
		return glm::translate(glm::mat4(1.0f), position) * glm::mat4_cast(orientation);
	}

	void aimAt(glm::vec3 target) {
		if(glm::length(target - position) < 1e-5f) return;
		auto direction = glm::normalize(target - position);
		auto up = std::abs(direction.z) > 0.99f ? glm::vec3(0, 1, 0) : glm::vec3(0, 0, 1);
		orientation = cameraNormalizeQuaternion(glm::quat_cast(glm::inverse(glm::lookAt(position, target, up))));
	}
};

struct CameraPath {
	std::vector<CameraKeyframe> keys;
	bool closed = false;
	bool smooth = true;

	int segmentCount() const {
		return keys.size() < 2 ? 0 : int(keys.size()) - (closed ? 0 : 1);
	}

	float duration() const {
		float seconds = 0;
		for(int i = 0; i < segmentCount(); ++i) seconds += keys[i].secondsToNext;
		return seconds;
	}

	float keyTime(int index) const {
		float seconds = 0;
		for(int i = 0; i < index && i < segmentCount(); ++i) seconds += keys[i].secondsToNext;
		return seconds;
	}

	CameraKeyframe sample(float seconds) const {
		if(keys.empty()) return {};
		if(keys.size() == 1) return keys.front();
		seconds = std::clamp(seconds, 0.0f, duration());
		int segment = 0;
		while(segment + 1 < segmentCount() && seconds > keys[segment].secondsToNext) {
			seconds -= keys[segment++].secondsToNext;
		}
		int next = (segment + 1) % int(keys.size());
		float u = std::clamp(seconds / std::max(keys[segment].secondsToNext, 0.01f), 0.0f, 1.0f);
		CameraKeyframe pose = keys[segment];
		pose.position = glm::mix(keys[segment].position, keys[next].position, u);
		if(smooth && keys.size() > 2) {
			const auto& p1 = keys[segment].position;
			const auto& p2 = keys[next].position;
			auto p0 = segment > 0 ? keys[segment - 1].position :
				(closed ? keys.back().position : 2.0f * p1 - p2);
			auto p3 = next + 1 < int(keys.size()) ? keys[next + 1].position :
				(closed ? keys.front().position : 2.0f * p2 - p1);
			// 向心 Catmull-Rom 减少间距不均时的过冲，重复位置退回线性插值。
			if(glm::length(p2 - p1) > 1e-5f) {
				float t0 = 0;
				float t1 = t0 + std::max(std::sqrt(glm::length(p1 - p0)), 1e-3f);
				float t2 = t1 + std::max(std::sqrt(glm::length(p2 - p1)), 1e-3f);
				float t3 = t2 + std::max(std::sqrt(glm::length(p3 - p2)), 1e-3f);
				float t = glm::mix(t1, t2, u);
				auto a1 = glm::mix(p0, p1, (t - t0) / (t1 - t0));
				auto a2 = glm::mix(p1, p2, (t - t1) / (t2 - t1));
				auto a3 = glm::mix(p2, p3, (t - t2) / (t3 - t2));
				auto b1 = glm::mix(a1, a2, (t - t0) / (t2 - t0));
				auto b2 = glm::mix(a2, a3, (t - t1) / (t3 - t1));
				pose.position = glm::mix(b1, b2, (t - t1) / (t2 - t1));
			}
		}
		// 四元数走最短旋转弧，避免跨越 180 度时突然翻转。
		pose.orientation = cameraSlerp(pose.orientation, keys[next].orientation, u);
		pose.fovy = glm::mix(keys[segment].fovy, keys[next].fovy, u);
		return pose;
	}
};
