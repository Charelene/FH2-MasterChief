"""Move the FH6 armour from FH6's driver skeleton (A-pose, long torso) onto the FH2 Stig skeleton (T-pose, short torso).

Two steps:
1. deform: every FH6 bone gets a transform T_i (FH6 bind space -> FH2 bind space); a vertex moves to
   sum_k w_k T_ik(v) (linear blend, the FH6 weights). Limbs: the FH6 segment joint->child is rotated onto the FH2
   segment and stretched along its axis to the FH2 length. Torso: a vertical piecewise-linear squeeze (hips, shoulder
   line, neck). Neck and head: one rigid move with the neck base, so the helmet keeps its shape and FH6's neck length.
2. reweight: FH6 bone weights become FH2 bone weights through BONE_MAP (twist and corrective bones fold into the FH2
   bone that carries that part of the limb, FH6's 3 spine bones spread over FH2's 5), top 4 kept, renormalized.
"""
import numpy as np

import geometry

SIDES = ("Left", "Right")

# FH6 bone -> {FH2 bone: share}. {S} = side. Unlisted FH6 bones fall back to their nearest mapped ancestor.
_SIDED = {
    "{S}UpLeg": {"{S}UpLeg": 1}, "{S}Leg": {"{S}Leg": 1}, "{S}Foot": {"{S}Foot": 1}, "{S}ToeBase": {"{S}ToeBase": 1},
    "{S}Pec_Corrective": {"Spine4": 1},
    "{S}Shoulder": {"{S}Shoulder": 1}, "{S}ShoulderHelper_Corrective": {"{S}Shoulder": 1},
    "{S}Arm": {"{S}Arm": 1}, "{S}Arm_TWIST0": {"{S}Arm": 1}, "{S}Arm_TWIST1": {"{S}Arm": 0.5, "{S}ArmRoll": 0.5},
    "{S}Arm_TWIST2": {"{S}ArmRoll": 1}, "{S}Arm_TWIST3": {"{S}ArmRoll": 1}, "{S}Bicep_Corrective": {"{S}ArmRoll": 1},
    "{S}Elbow_Corrective": {"{S}ArmRoll": 0.5, "{S}Forearm": 0.5}, "{S}Elbow_CorrectiveBack": {"{S}ArmRoll": 0.5, "{S}Forearm": 0.5},
    "{S}Elbow_CorrectiveFront": {"{S}ArmRoll": 0.5, "{S}Forearm": 0.5},
    "{S}ForeArm": {"{S}Forearm": 1}, "{S}ForeArm_TWIST0": {"{S}Forearm": 1}, "{S}ForeArm_TWIST1": {"{S}Forearm": 0.5, "{S}ForarmRoll": 0.5},
    "{S}ForeArm_TWIST2": {"{S}ForarmRoll": 1}, "{S}ForeArm_TWIST3": {"{S}ForarmRoll": 1},
    "{S}Wrist_Corrective": {"{S}ForarmRoll": 1}, "{S}WristDown_Corrective": {"{S}ForarmRoll": 1}, "{S}WristUp_Corrective": {"{S}FingerBase": 1},
    "{S}Hand": {"{S}FingerBase": 1}, "{S}Hand_PROP": {"{S}FingerBase": 1},
    "{S}IndexMeta": {"{S}FingerBase": 1}, "{S}MiddleMeta": {"{S}FingerBase": 1}, "{S}RingMeta": {"{S}FingerBase": 1}, "{S}PinkyMeta": {"{S}FingerBase": 1},
}
for _f in ("Index", "Middle", "Ring", "Pinky", "Thumb"):
    for _n in (1, 2, 3):
        _SIDED["{S}%s%d" % (_f, _n)] = {"{S}Hand%s%d" % (_f, _n): 1}

BONE_MAP = {"<root>": {"Hips": 1}, "Root": {"Hips": 1}, "Hips": {"Hips": 1},
            "Spine": {"Spine": 0.5, "Spine1": 0.5}, "Spine1": {"Spine2": 0.5, "Spine3": 0.5}, "Spine2": {"Spine4": 1},
            "Neck": {"Neck": 1}, "Neck1": {"Neck1": 1}, "Head": {"Head": 1}}
for _k, _v in _SIDED.items():
    for _s in SIDES:
        BONE_MAP[_k.replace("{S}", _s)] = {b.replace("{S}", _s): w for b, w in _v.items()}

# deformation groups: FH6 bone -> how it moves. ("torso",) | ("rigid", fh6 joint, fh2 joint)
#                                               | ("segment", fh6 a, fh6 b, fh2 a, fh2 b, stretch, lateral)
_SEG = {
    "{S}UpLeg": ("{S}UpLeg", "{S}Leg", "{S}UpLeg", "{S}Leg"), "{S}Leg": ("{S}Leg", "{S}Foot", "{S}Leg", "{S}Foot"),
    "{S}Foot": ("{S}Foot", "{S}ToeBase", "{S}Foot", "{S}ToeBase"), "{S}ToeBase": ("{S}Foot", "{S}ToeBase", "{S}Foot", "{S}ToeBase"),
    "{S}Shoulder": ("{S}Shoulder", "{S}Arm", "{S}Shoulder", "{S}Arm"), "{S}ShoulderHelper_Corrective": ("{S}Shoulder", "{S}Arm", "{S}Shoulder", "{S}Arm"),
    "{S}Arm": ("{S}Arm", "{S}ForeArm", "{S}Arm", "{S}Forearm"), "{S}ForeArm": ("{S}ForeArm", "{S}Hand", "{S}Forearm", "{S}FingerBase"),
    "{S}Hand": ("{S}Hand", "{S}Middle1", "{S}FingerBase", "{S}HandMiddle1"),
}
for _f in ("Index", "Middle", "Ring", "Pinky"):
    _SEG["{S}%s1" % _f] = ("{S}%s1" % _f, "{S}%s2" % _f, "{S}Hand%s1" % _f, "{S}Hand%s2" % _f)
    _SEG["{S}%s2" % _f] = ("{S}%s2" % _f, "{S}%s3" % _f, "{S}Hand%s2" % _f, "{S}Hand%s3" % _f)
    _SEG["{S}%s3" % _f] = ("{S}%s2" % _f, "{S}%s3" % _f, "{S}Hand%s2" % _f, "{S}Hand%s3" % _f)
_SEG["{S}Thumb1"] = ("{S}Thumb1", "{S}Thumb2", "{S}HandThumb1", "{S}HandThumb2")
_SEG["{S}Thumb2"] = ("{S}Thumb2", "{S}Thumb3", "{S}HandThumb2", "{S}HandThumb3")
_SEG["{S}Thumb3"] = ("{S}Thumb2", "{S}Thumb3", "{S}HandThumb2", "{S}HandThumb3")

STRETCH_LIMITS = (0.75, 1.35)


def _wpos(world, skel, name):
    i = skel.index(name)
    if i < 0:
        raise KeyError(name)
    return world[i][3, :3].copy()


def _rot_between(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def _frame(d, lat):
    d = d / np.linalg.norm(d)
    lat = lat - d * np.dot(lat, d)
    lat = lat / np.linalg.norm(lat)
    return np.stack([d, lat, np.cross(d, lat)])  # rows


class Retarget:
    def __init__(self, skel6, skel2):
        self.s6, self.s2 = skel6, skel2
        self.w6, self.w2 = geometry.world_matrices(skel6), geometry.world_matrices(skel2)
        self.fh2_index = {b.name: i for i, b in enumerate(skel2.bones)}
        self._build_transforms()

    # ------------------------------------------------------------------ transforms per FH6 bone
    def p6(self, n):
        return _wpos(self.w6, self.s6, n)

    def p2(self, n):
        return _wpos(self.w2, self.s2, n)

    def _torso_y(self, y):
        """FH6 height -> FH2 height on the torso: hips, shoulder line and neck land on FH2's; below the hips and above
        the neck heights shift without scaling."""
        k6 = [self.p6("Hips")[1], (self.p6("LeftArm")[1] + self.p6("RightArm")[1]) / 2, self.p6("Neck")[1]]
        k2 = [self.p2("Hips")[1], (self.p2("LeftArm")[1] + self.p2("RightArm")[1]) / 2, self.p2("Neck")[1]]
        y = np.asarray(y, dtype=np.float64)
        out = np.interp(y, k6, k2)
        out = np.where(y < k6[0], y + (k2[0] - k6[0]), out)
        out = np.where(y > k6[2], y + (k2[2] - k6[2]), out)
        return out

    def _build_transforms(self):
        self.kind = {}
        for i, b in enumerate(self.s6.bones):
            self.kind[i] = self._transform_for(b.name, i)

    def _transform_for(self, name, i):
        for s in SIDES:
            for k, seg in _SEG.items():
                if k.replace("{S}", s) == name:
                    a6, b6, a2, b2 = (x.replace("{S}", s) for x in seg)
                    return self._segment(a6, b6, a2, b2, finger=any(f in name for f in ("Index", "Middle", "Ring", "Pinky", "Thumb", "Hand")), side=s)
        # neck and helmet move as one piece with the neck base: FH2's driver has a 7 cm shorter neck than FH6's, and
        # moving the helmet onto FH2's head joint sank its chin into the chest plate (seen in game 2026-11-13)
        if name in ("Head", "Jaw", "Neck1", "Neck") or self._under(i, "Head"):
            return ("rigid", self.p6("Neck"), self.p2("Neck"))
        # anything else inherits from its parent chain, torso by default
        parent = self.s6.bones[i].parent
        while parent >= 0:
            pn = self.s6.bones[parent].name
            if pn in ("Spine", "Spine1", "Spine2", "Hips", "Root", "<root>"):
                return ("torso",)
            if parent in self.kind:
                return self.kind[parent]
            parent = self.s6.bones[parent].parent
        return ("torso",)

    def _under(self, i, ancestor):
        p = self.s6.bones[i].parent
        while p >= 0:
            if self.s6.bones[p].name == ancestor:
                return True
            p = self.s6.bones[p].parent
        return False

    def _segment(self, a6, b6, a2, b2, finger, side):
        A6, B6, A2, B2 = self.p6(a6), self.p6(b6), self.p2(a2), self.p2(b2)
        d6, d2 = B6 - A6, B2 - A2
        hand_lat6 = self.p6(side + "Pinky1") - self.p6(side + "Index1")
        hand_lat2 = self.p2(side + "HandPinky1") - self.p2(side + "HandIndex1")
        if finger or "Arm" in a6 and "Fore" in a6:
            # forearm, hand and fingers: align the bone and the palm's index->pinky axis (no wrist twist)
            R = np.linalg.solve(_frame(d6, hand_lat6), _frame(d2, hand_lat2))  # rows: R maps row vectors v6 -> v2
        else:
            R = _rot_between(d6, d2).T  # row-vector form
        s = np.clip(np.linalg.norm(d2) / max(np.linalg.norm(d6), 1e-9), *STRETCH_LIMITS)
        return ("segment", A6, d6 / np.linalg.norm(d6), s, R, A2)

    def apply(self, kind, v):
        """v: (N,3) FH6 bind positions -> (N,3) FH2 bind positions for one transform; also returns the rotation."""
        if kind[0] == "torso":
            out = v.copy()
            out[:, 1] = self._torso_y(v[:, 1])
            out[:, 2] += self.p2("Spine4")[2] - self.p6("Spine2")[2]
            return out, np.eye(3)
        if kind[0] == "rigid":
            return v + (kind[2] - kind[1]), np.eye(3)
        _, A6, d6, s, R, A2 = kind
        rel = v - A6
        along = rel @ d6
        rel = rel + np.outer(along * (s - 1.0), d6)
        return rel @ R + A2, R

    # ------------------------------------------------------------------ per part
    def deform(self, positions, normals, tangents, joints, weights):
        out_p = np.zeros_like(positions, dtype=np.float64)
        out_n = np.zeros_like(normals, dtype=np.float64)
        out_t = np.zeros((len(positions), 3)) if tangents is not None else None
        uniq = np.unique(joints[weights > 0])
        for b in uniq:
            sel = np.any((joints == b) & (weights > 0), axis=1)
            w = np.sum(np.where(joints[sel] == b, weights[sel], 0), axis=1)[:, None]
            p, R = self.apply(self.kind[int(b)], positions[sel].astype(np.float64))
            out_p[sel] += w * p
            out_n[sel] += w * (normals[sel] @ R)
            if out_t is not None:
                out_t[sel] += w * (tangents[sel, :3] @ R)
        out_n /= np.maximum(np.linalg.norm(out_n, axis=1, keepdims=True), 1e-12)
        if out_t is not None:
            out_t -= out_n * np.sum(out_t * out_n, axis=1, keepdims=True)
            out_t /= np.maximum(np.linalg.norm(out_t, axis=1, keepdims=True), 1e-12)
            out_t = np.concatenate([out_t, tangents[:, 3:4]], axis=1)
        return out_p.astype(np.float32), out_n.astype(np.float32), None if out_t is None else out_t.astype(np.float32)

    def reweight(self, joints, weights, max_influences=4):
        n = len(joints)
        acc = [dict() for _ in range(n)]
        for k in range(joints.shape[1]):
            for vi in np.nonzero(weights[:, k] > 0)[0]:
                for fh2, share in self._map(int(joints[vi, k])).items():
                    acc[vi][fh2] = acc[vi].get(fh2, 0.0) + float(weights[vi, k]) * share
        J = np.zeros((n, max_influences), np.int64)
        W = np.zeros((n, max_influences), np.float32)
        for vi, d in enumerate(acc):
            top = sorted(d.items(), key=lambda kv: -kv[1])[:max_influences]
            tot = sum(w for _, w in top) or 1.0
            for k, (j, w) in enumerate(top):
                J[vi, k], W[vi, k] = j, w / tot
        return J, W

    def _map(self, i6):
        i = i6
        while i >= 0:
            name = self.s6.bones[i].name
            if name in BONE_MAP:
                return {self.fh2_index[b]: s for b, s in BONE_MAP[name].items()}
            if name.startswith(("L_", "R_", "M_", "Nose", "Tongue", "Jaw")) or "Brow" in name or "Eye" in name:
                return {self.fh2_index["Head"]: 1.0}
            i = self.s6.bones[i].parent
        return {self.fh2_index["Hips"]: 1.0}
