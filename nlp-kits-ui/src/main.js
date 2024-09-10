/**
 * main.js
 *
 * Bootstraps Vuetify and other plugins then mounts the App`
 */

// styles
import "@/assets/css/global.scss";

// Plugins
import { registerPlugins } from "@/plugins";

// Components
import App from "./App.vue";

// Composables
import { createApp } from "vue";

const app = createApp(App);

registerPlugins(app);

app.mount("#app");
