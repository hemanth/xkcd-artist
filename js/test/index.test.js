import test from 'node:test';
import assert from 'node:assert/strict';
import xkcdArtist from '../src/index.js';

test('xkcdArtist synthesizes multi-panel XKCD comic with hand-lettered speech bubbles and alt text', () => {
  const comic = xkcdArtist('Sudo make me a sandwich');
  assert.equal(comic.is_invalid, false);
  assert.equal(comic.title, 'SANDWICH');
  assert.ok(comic.panels.length >= 2);
  assert.ok(comic.alt.includes('Simon Says'));

  const svg = comic.toSVG();
  assert.ok(svg.startsWith('<svg'));
  assert.ok(svg.includes('SUDO MAKE ME A SANDWICH'));
  assert.ok(svg.includes('Simon Says'));
});

test('xkcdArtist handles arbitrary developer prompts and abstains on empty/garbage inputs', () => {
  const custom = xkcdArtist('Cueball debugging a Kubernetes YAML file at 3am with Ponytail');
  assert.equal(custom.is_invalid, false);
  assert.ok(custom.confidence > 0.99);
  assert.ok(custom.toSVG().includes('<path'));

  const invalid = xkcdArtist('   ');
  assert.equal(invalid.is_invalid, true);
  assert.ok(invalid.p_invalid > 0.99);
});
