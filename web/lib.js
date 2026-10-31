// Pure helpers for the demo page. No DOM access, so they can be tested with `node --test`.

/** Sort a copy of `users` by predicted uplift key, highest first. key: 'uplift_x' | 'uplift_t' | 'true_effect'. */
function rankBy(users, key) {
  return [...users].sort((a, b) => b[key] - a[key] || a.id - b.id);
}

/**
 * Expected extra retained users if the top `n` users by `key` are contacted.
 * Uses the simulation's true per-user effect (ground truth, only known because the data is simulated).
 */
function expectedGain(users, key, n) {
  if (!Number.isInteger(n) || n < 0 || n > users.length) {
    throw new RangeError(`n must be an integer between 0 and ${users.length}`);
  }
  const ranked = rankBy(users, key);
  let sum = 0;
  for (let i = 0; i < n; i++) sum += ranked[i].true_effect;
  return sum;
}

/** Random targeting: every user is equally likely to be picked, so the expected gain is n * mean effect. */
function randomGain(users, n) {
  if (!users.length) return 0;
  const mean = users.reduce((s, u) => s + u.true_effect, 0) / users.length;
  return n * mean;
}

/** Gain at every n from 0..users.length for a ranking key; used for the curve. */
function gainCurve(users, key, points = 40) {
  const ranked = rankBy(users, key);
  const out = [];
  let sum = 0;
  let next = 0;
  for (let i = 0; i <= ranked.length; i++) {
    if (i > 0) sum += ranked[i - 1].true_effect;
    if (i >= next || i === ranked.length) {
      out.push([i, sum]);
      next = i + Math.max(1, Math.floor(ranked.length / points));
    }
  }
  return out;
}

/** Parse a user-id string typed by a person. Returns {ok, user} or {ok:false, error}. */
function lookupUser(users, text) {
  const t = String(text).trim();
  if (t === '') return { ok: false, error: 'Enter a user id.' };
  if (!/^\d+$/.test(t)) return { ok: false, error: 'A user id is a whole number.' };
  const id = Number(t);
  const user = users.find((u) => u.id === id);
  if (!user) return { ok: false, error: `User ${id} is not in the held-out set (the page only shows users the models did not train on).` };
  return { ok: true, user };
}

const lib = { rankBy, expectedGain, randomGain, gainCurve, lookupUser };
if (typeof module !== 'undefined') module.exports = lib;
if (typeof window !== 'undefined') window.demoLib = lib;
