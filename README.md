# BeZeus

Camera-controlled lightning prototype built with Next.js, MediaPipe hand tracking, WebGL, and Framer Motion.

## Product Scope

BeZeus turns a simple gesture into a live visual effect:

- request camera permission from the browser
- detect a single hand with MediaPipe
- map palm position, openness, movement, and angle into lightning output
- keep a settings screen available for status recovery and failure diagnosis

## Key Requirements

- The first screen must explain the camera-based interaction before asking for permission.
- Permission denial, missing devices, WebGL failure, and model startup errors must show different user-facing messages.
- The interaction must remain inspectable from the settings screen even when the camera is not active.
- Motion-heavy effects must respect reduced-motion preferences.
- Manual QA scenarios must be visible enough for a reviewer to reproduce the main states.

## Local Checks

```bash
npm run lint
npm run build
```

## Manual QA Checklist

- Open the app and confirm the prepare dialog explains why the camera is needed.
- Click `Enable Camera to be Zeus`, allow the camera, and confirm the live status changes to ready.
- Open and close the hand to confirm the visible state changes between closed and open palm.
- Deny camera permission in browser settings and confirm the recovery message is specific.
- Open the `Setting` view and verify camera, hand, and gesture states are listed separately.
- Enable reduced-motion at the OS/browser level and confirm large motion effects are reduced.

## Implementation Notes

The UI treats camera tracking as a stateful interaction rather than a one-shot visual trick. The central component keeps camera state, stream cleanup, hand landmark analysis, WebGL fallback, status copy, and visible test scenarios close to the interaction they describe.
