const test = require('node:test');
const assert = require('node:assert');
const lib = require('../web/lib.js');

const users = [
  { id: 1, true_effect: 0.1, uplift_x: 0.3, uplift_t: 0.1 },
  { id: 2, true_effect: 0.4, uplift_x: 0.2, uplift_t: 0.5 },
  { id: 3, true_effect: -0.1, uplift_x: 0.9, uplift_t: 0.0 },
  { id: 4, true_effect: 0.3, uplift_x: 0.1, uplift_t: 0.2 },
];

test('rankBy sorts highest first without mutating input', () => {
  const r = lib.rankBy(users, 'uplift_x');
  assert.deepStrictEqual(r.map((u) => u.id), [3, 1, 2, 4]);
  assert.strictEqual(users[0].id, 1);
});

test('expectedGain sums true effects of the top n', () => {
  assert.strictEqual(lib.expectedGain(users, 'uplift_x', 2), 0.0);
  assert.ok(Math.abs(lib.expectedGain(users, 'true_effect', 2) - 0.7) < 1e-9);
  assert.strictEqual(lib.expectedGain(users, 'uplift_x', 0), 0);
});

test('expectedGain rejects out-of-range n', () => {
  assert.throws(() => lib.expectedGain(users, 'uplift_x', -1), RangeError);
  assert.throws(() => lib.expectedGain(users, 'uplift_x', 5), RangeError);
  assert.throws(() => lib.expectedGain(users, 'uplift_x', 1.5), RangeError);
});

test('the oracle ranking is never worse than random at any n', () => {
  for (let n = 0; n <= users.length; n++) {
    assert.ok(lib.expectedGain(users, 'true_effect', n) >= lib.randomGain(users, n) - 1e-9);
  }
});

test('randomGain equals total effect when everyone is contacted', () => {
  const total = users.reduce((s, u) => s + u.true_effect, 0);
  assert.ok(Math.abs(lib.randomGain(users, users.length) - total) < 1e-9);
});

test('gainCurve starts at 0 and ends at the full sum', () => {
  const c = lib.gainCurve(users, 'uplift_x', 2);
  assert.deepStrictEqual(c[0], [0, 0]);
  assert.strictEqual(c[c.length - 1][0], users.length);
});

test('lookupUser validates input', () => {
  assert.strictEqual(lib.lookupUser(users, '').ok, false);
  assert.strictEqual(lib.lookupUser(users, 'abc').ok, false);
  assert.strictEqual(lib.lookupUser(users, '-3').ok, false);
  assert.strictEqual(lib.lookupUser(users, '99').ok, false);
  const r = lib.lookupUser(users, ' 2 ');
  assert.strictEqual(r.ok, true);
  assert.strictEqual(r.user.id, 2);
});
